#!/usr/bin/env python3
"""
Offline visszajátszás a TELJES vision pipeline-on (ChessVisionTracker, valódi
modell, valódi detektor) — lépés-elfogadási latencia és hamis elfogadás mérése
kamera nélkül, determinisztikus órával.

Két forrás:

1. `sessions` — a FEN-gyűjtő mentett képei (data_fen/frames + shots.jsonl).
   Egy sessionön belül a szomszédos fotók egy legális lépésnyire vannak
   egymástól (positions.txt), tehát a sorozat egy "játszma": az első képet
   kezdőállásként adjuk a trackernek, majd minden képet `--hold` másodpercig
   tartunk 30 fps-en (ugyanazt a JPEG-et ismételve, opcionális kamera-zajjal),
   közben a képek között `--gap` másodpercnyi "kéz" fázist szimulálunk
   (szintetikus takarás + mozgás a két állás különbség-mezői körül).
   Mérjük: minden lépésnél hány frame / mennyi szimulált idő telt el a végállapot
   első frame-jétől az elfogadásig, a lépés helyes-e (a FEN-pár lépése), és
   volt-e HAMIS elfogadás (bármely olyan lépés, ami nem a valós lépés) vagy
   BERAGADÁS (nem fogadta el a lépést a hold ideje alatt).

   FIGYELEM: statikus fotók ismétlése optimista (nincs valódi villódzás, nincs
   valódi kéz). A kéz-fázis szintetikus. Valódi játék méréséhez:
   tools/record_camera.py felvétel + `video` mód.

2. `video` — tools/record_camera.py felvétele (frame-ek + timestamps.csv) vagy
   bármilyen videó (akkor a fps-ből számolt idő). Kiírja az elfogadott lépéseket
   időbélyeggel; --pgn / --moves megadásával ellenőrzi a sorrendet.

Használat (repo gyökérből):

    python -m tools.replay_frames sessions
    python -m tools.replay_frames sessions --sessions jo_fenyviszony_1080 --hold 1.5 --gap 0.8
    python -m tools.replay_frames video felvetel_dir/            # record_camera.py kimenete
    python -m tools.replay_frames video jatszma.avi --moves "e2e4 e7e5 g1f3"
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import chess
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.app.config import AppConfig, STABILIZER_PARAMS  # noqa: E402
from vision.pipeline.tracker import ChessVisionTracker  # noqa: E402

DT = 1 / 30.0


class SimClock:
    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t


# ---------------------------------------------------------------------------
# sessions mód
# ---------------------------------------------------------------------------

def _legal_move_between(fen_a: str, fen_b: str) -> str | None:
    a = chess.Board(fen_a)
    target = chess.Board(fen_b).board_fen()
    for mv in a.legal_moves:
        b = a.copy()
        b.push(mv)
        if b.board_fen() == target:
            return mv.uci()
    return None


def _sequences(shots: list[dict]) -> list[list[dict]]:
    """Sessionönként a leghosszabb egymás utáni, egy-egy legális lépésnyire lévő futamok."""
    out: list[list[dict]] = []
    cur: list[dict] = []
    for s in shots:
        if cur and s["session"] == cur[-1]["session"] and _legal_move_between(cur[-1]["fen"], s["fen"]):
            cur.append(s)
        else:
            if len(cur) >= 2:
                out.append(cur)
            cur = [s]
    if len(cur) >= 2:
        out.append(cur)
    return out


def _noisy(frame: np.ndarray, rng: np.random.Generator, amp: int) -> np.ndarray:
    if amp <= 0:
        return frame
    n = rng.integers(-amp, amp + 1, frame.shape, dtype=np.int16)
    return np.clip(frame.astype(np.int16) + n, 0, 255).astype(np.uint8)


def _hand_frame(frame_from: np.ndarray, frame_to: np.ndarray, tracker: ChessVisionTracker, phase: float,
                rng: np.random.Generator) -> np.ndarray:
    """Szintetikus kéz-fázis: a két kép közti átmenet, egy mozgó sötét folttal a
    táblán (a warp-koordináták helyett a képen; a cél csak az, hogy legyen mozgás
    és takarás a tábla felett — a pontos alak nem számít)."""
    img = frame_from if phase < 0.5 else frame_to
    img = img.copy()
    det = tracker.det
    if det is not None and det.centers_img is not None:
        cs = np.array([det.centers_img[r][c] for r in range(8) for c in range(8)], dtype=np.float32)
        cx, cy = cs.mean(axis=0)
        span = (cs.max(axis=0) - cs.min(axis=0)).max()
        # a folt a tábla közepe körül köröz
        ang = phase * 2 * np.pi
        px = int(cx + 0.35 * span * np.cos(ang))
        py = int(cy + 0.35 * span * np.sin(ang))
        rad = int(max(20, span * 0.12))
        cv2.circle(img, (px, py), rad, (40, 30, 30), -1)
        cv2.circle(img, (px + rad, py), rad // 2, (60, 50, 50), -1)
    return _noisy(img, rng, 2)


def run_sessions(args: argparse.Namespace) -> None:
    shots = [json.loads(l) for l in open(args.shots, encoding="utf-8") if l.strip()]
    if args.sessions:
        shots = [s for s in shots if s["session"] in set(args.sessions)]
    seqs = _sequences(shots)
    print(f"[replay] {len(shots)} kép -> {len(seqs)} futam (szomszédos képek egy legális lépésnyire): "
          + ", ".join(f"{q[0]['session']}x{len(q)}" for q in seqs))

    cfg = AppConfig()
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    n_false = n_stuck = n_ok = 0
    t_wall = time.perf_counter()

    for seq in seqs:
        clock = SimClock()
        cfg_seq = AppConfig(**{**cfg.__dict__, "start_fen": seq[0]["fen"], "enable_pipeline_profiler": True})
        tracker = ChessVisionTracker(cfg_seq, clock=clock)
        frames = [cv2.imread(str(args.frames_dir / f"{s['base']}.jpg")) for s in seq]
        if any(f is None for f in frames):
            print("  ! hiányzó kép a futamban, kihagyva")
            continue

        # init
        ok_init = False
        for _ in range(60):
            res = tracker.process_frame(_noisy(frames[0], rng, args.noise))
            clock.t += DT
            if tracker.initialized:
                ok_init = True
                break
        if not ok_init:
            print(f"  {seq[0]['session']}: init sikertelen ({res.mode}) — futam kihagyva")
            n_stuck += len(seq) - 1
            continue
        # hold az első képen
        for _ in range(int(args.hold / DT)):
            tracker.process_frame(_noisy(frames[0], rng, args.noise))
            clock.t += DT

        for i in range(1, len(seq)):
            truth = _legal_move_between(seq[i - 1]["fen"], seq[i]["fen"])
            expected_ply = len(tracker.game.move_history)
            # kéz-fázis
            n_gap = int(args.gap / DT)
            for k in range(n_gap):
                res = tracker.process_frame(_hand_frame(frames[i - 1], frames[i], tracker, k / max(1, n_gap), rng))
                clock.t += DT
                if res.board_changed:
                    n_false += 1
                    rows.append({"session": seq[0]["session"], "ply": i, "truth": truth, "got": res.uci,
                                 "phase": "hand", "frames": k, "latency_ms": "", "mode": res.mode, "ok": False})
                    print(f"  !! HAMIS elfogadás a kéz-fázisban: {res.uci} (valós: {truth}) [{res.mode}]")
            # végállapot
            t_state = clock.t
            accepted = None
            n_hold = int(args.hold / DT)
            modes: dict[str, int] = {}
            for k in range(n_hold):
                res = tracker.process_frame(_noisy(frames[i], rng, args.noise))
                clock.t += DT
                modes[res.mode] = modes.get(res.mode, 0) + 1
                if res.board_changed:
                    accepted = (k + 1, res)
                    break
            if accepted is None:
                n_stuck += 1
                top = sorted(modes.items(), key=lambda kv: -kv[1])[:3]
                rows.append({"session": seq[0]["session"], "ply": i, "truth": truth, "got": "", "phase": "hold",
                             "frames": n_hold, "latency_ms": "", "mode": " ".join(f"{m}:{n}" for m, n in top), "ok": False})
                print(f"  -- BERAGADÁS {seq[0]['session']} ply {i}: {truth} nem lett elfogadva {args.hold:.1f} s alatt; "
                      f"módok: {top}")
                # a játszma folytatásához az igazságot alkalmazzuk
                tracker.game.apply_uci(truth)
                tracker.accepted_occ = np.asarray(chess_occ(seq[i]["fen"]), np.int32)
                tracker.stabilizer.set_reference(tracker.accepted_occ.tolist())
                continue
            k, res = accepted
            lat = (res.accept_info["t_accept"] - t_state) * 1000.0
            ok = res.uci == truth
            if not ok:
                n_false += 1
                print(f"  !! HAMIS elfogadás: {res.uci} (valós: {truth}) [{res.mode}] {seq[0]['session']} ply {i}")
                # visszaállítjuk az igazságra, hogy a futam folytatható legyen
                tracker.game = type(tracker.game)(seq[i]["fen"])
                tracker.accepted_occ = np.asarray(chess_occ(seq[i]["fen"]), np.int32)
                tracker.stabilizer.set_reference(tracker.accepted_occ.tolist())
            else:
                n_ok += 1
            rows.append({"session": seq[0]["session"], "ply": i, "truth": truth, "got": res.uci, "phase": "hold",
                         "frames": k, "latency_ms": f"{lat:.1f}", "mode": res.mode, "ok": ok})
        if tracker.profiler:
            st = tracker.profiler.get_stats()
            keys = ("board_detect", "classifier_full", "classifier_partial", "square_diff", "stabilizer", "resolve")
            print(f"  {seq[0]['session']}: " + "  ".join(f"{k}={st[k]['p50_ms']:.1f}ms" for k in keys if k in st)
                  + f"  (redetect: {tracker.redetect_count}, init igazítás: {tracker.init_alignment})")

    lats = [float(r["latency_ms"]) for r in rows if r["latency_ms"]]
    print("\n" + "=" * 90)
    print(f"  VISSZAJÁTSZÁS: {n_ok} helyes elfogadás, {n_false} HAMIS elfogadás, {n_stuck} beragadás "
          f"({len(rows)} lépés, {time.perf_counter() - t_wall:.0f} s)")
    if lats:
        lats.sort()
        print(f"  latencia a végállapot első frame-jétől (szimulált 30 fps): átlag {statistics.mean(lats):.0f} | "
              f"p50 {lats[len(lats) // 2]:.0f} | p95 {lats[min(int(len(lats) * 0.95), len(lats) - 1)]:.0f} | max {lats[-1]:.0f} ms")
        print(f"  (elméleti minimum: min_stable_s={STABILIZER_PARAMS.min_stable_s * 1000:.0f} ms, "
              f"min_static_s={STABILIZER_PARAMS.min_static_s * 1000:.0f} ms a kéz eltűnésétől)")
    print("=" * 90)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["session", "ply", "truth", "got", "phase", "frames", "latency_ms", "mode", "ok"])
            w.writeheader()
            w.writerows(rows)
        print(f"  részletek: {args.out}")


def chess_occ(fen: str):
    from chess_logic import board_to_occupancy
    return board_to_occupancy(chess.Board(fen))


# ---------------------------------------------------------------------------
# video mód
# ---------------------------------------------------------------------------

def _iter_recording(path: Path):
    """(t, frame) párok: record_camera.py könyvtár (frames/*.jpg + timestamps.csv) vagy videófájl."""
    if path.is_dir():
        ts = list(csv.DictReader(open(path / "timestamps.csv", encoding="utf-8")))
        for row in ts:
            frame = cv2.imread(str(path / "frames" / row["file"]))
            if frame is not None:
                yield float(row["t"]), frame
        return
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i / fps, frame
        i += 1
    cap.release()


def run_video(args: argparse.Namespace) -> None:
    cfg = AppConfig(start_fen=args.fen) if args.fen else AppConfig()
    clock = SimClock()
    tracker = ChessVisionTracker(cfg, clock=clock)
    expected = args.moves.split() if args.moves else []
    accepted: list[tuple[float, str, str]] = []
    n = 0
    t_first = None
    last_motion_stop = None
    for t, frame in _iter_recording(args.video):
        if t_first is None:
            t_first = t
        clock.t = 1000.0 + (t - t_first)
        res = tracker.process_frame(frame)
        n += 1
        if res.board_changed:
            info = res.accept_info or {}
            lat_state = (info["t_accept"] - info["candidate_since"]) * 1000 if info.get("candidate_since") else float("nan")
            lat_static = (info["t_accept"] - info["last_motion_t"]) * 1000 if info.get("last_motion_t") else float("nan")
            accepted.append((t - t_first, res.uci, res.mode))
            idx = len(accepted) - 1
            exp = expected[idx] if idx < len(expected) else None
            flag = "" if exp is None else ("  OK" if exp == res.uci else f"  !! HAMIS (várt: {exp})")
            print(f"  t={t - t_first:7.2f}s  {res.uci}  [{res.mode}]  latencia: végállapottól {lat_state:.0f} ms, "
                  f"utolsó mozgástól {lat_static:.0f} ms{flag}")
    print(f"\n[video] {n} frame, {len(accepted)} elfogadott lépés, redetect: {tracker.redetect_count}")
    if expected:
        got = [u for _, u, _ in accepted]
        print(f"  várt: {' '.join(expected)}\n  kapott: {' '.join(got)}\n  egyezés: {got == expected}")
    if tracker.profiler:
        tracker.profiler.report()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    ps = sub.add_parser("sessions")
    ps.add_argument("--frames-dir", type=Path, default=ROOT / "data_fen" / "frames")
    ps.add_argument("--shots", type=Path, default=ROOT / "data_fen" / "shots.jsonl")
    ps.add_argument("--sessions", nargs="*", default=None)
    ps.add_argument("--hold", type=float, default=2.0, help="másodperc, ameddig egy állást tartunk (30 fps)")
    ps.add_argument("--gap", type=float, default=0.8, help="szintetikus kéz-fázis hossza két állás között")
    ps.add_argument("--noise", type=int, default=2, help="+-N szintetikus pixelzaj az ismételt képeken")
    ps.add_argument("--out", type=Path, default=None, help="részletes CSV")

    pv = sub.add_parser("video")
    pv.add_argument("video", type=Path, help="record_camera.py könyvtár vagy videófájl")
    pv.add_argument("--fen", default=None)
    pv.add_argument("--moves", default=None, help="várt lépések UCI-ban, szóközzel")

    args = p.parse_args()
    logging_setup()
    if args.mode == "sessions":
        run_sessions(args)
    else:
        run_video(args)


def logging_setup():
    import logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")


if __name__ == "__main__":
    main()
