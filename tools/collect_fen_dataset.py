"""
Tanítóadat-gyűjtő ISMERT FEN-ből — automatikus, zajmentes címkézés.

Elv: a felhasználó felállítja a táblát egy ismert állásra és megadja a FEN-t.
A szkript ebből mind a 64 mezőt címkézi (foglaltság + bábutípus), kézi
címkézés és címkézési zaj nélkül. A kivágás PONTOSAN az éles pipeline útja:

    detect_board_on_frame(gray)  ->  warpPerspective  ->  crop_with_context(context=0.50)

(ugyanaz, mint tools/dump_live_rois.py és a ChessVisionTracker), a címke pedig
FEN -> chess_logic.resolver.board_to_occupancy + python-chess -> a nyers rács
(tools/fen_labels.py, a pipeline raw_to_standard-jának inverzével).

Kimenet (kompatibilis a meglévő train_new/val_new/{black,empty,white} szerkezettel):

    <out-dir>/train_new/{black,empty,white}/<session>_<ts>_r<r>c<c>_<mező>_<bábu>.jpg
    <out-dir>/val_new/{black,empty,white}/...
    <out-dir>/frames/<session>_<ts>.jpg     teljes kamerakép (később más context-tel újravágható)
    <out-dir>/labels.csv                    minden ROI: fájl, split, FEN, mező, szín, típus, exponálás...
    <out-dir>/shots.jsonl                   fotónkénti meta (FEN, kamera-beállítás, egyezés a modellel)

A fájlnév végén a `_<mező>_<bábu>` tag hordozza a bábutípust (`P`,`n`,... vagy
`x` = üres); a tanítószkript (tools/train_square_classifier.py) ebből olvassa
a típus-fej címkéjét. A régi, tag nélküli képekkel együtt használható.

Használat (repo gyökérből):

    python -m tools.collect_fen_dataset --session nappali_de --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    python -m tools.collect_fen_dataset --session este_lampa --fen-file positions.txt --burst 3
    python -m tools.collect_fen_dataset --session ... --exposure 250 --wb-temp 4600   # fix exponálás

Billentyűk az előnézeti ablakban:
    SPACE  fotó (--burst darab frame) -> 64 ROI mentése a FEN címkéivel
    n      következő FEN (--fen-file listából; ha nincs, a terminálban kér be)
    f      FEN bekérése a terminálban
    d      tábla újradetektálása (homográfia) — ha a kamera/tábla elmozdult
    o      címke-overlay ki/be a warpolt táblán
    e      kamera exponálás / fehéregyensúly állapot kiírása
    q      kilépés

FONTOS — FIX EXPONÁLÁS ÉS FEHÉREGYENSÚLY:
    Auto-exposure mellett a fekete bábu látszó fényereje frame-ről frame-re
    változik (a kamera a világos táblához igazít), ezért a gyűjtés ELŐTT
    rögzítsd az exponálást és a fehéregyensúlyt (--exposure / --wb-temp, vagy
    v4l2-ctl), és ugyanezt használd élesben is. A szkript figyeli a tábla
    átlagfényességének ingadozását, és figyelmeztet, ha az auto-exposure
    valószínűleg még be van kapcsolva.

Sorozatgyűjtés: több állás (--fen-file), több fényviszony és napszak — minden
fényviszonyhoz KÜLÖN --session nevet adj (pl. `delelott_ablak`, `este_lampa`),
így a fájlnevekből visszakereshető, melyik körülmény hiányzik az adatból.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path

import chess
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fen_labels import SquareLabel, fen_to_raw_labels, fen_to_raw_occupancy  # noqa: E402
from vision.app.config import AppConfig  # noqa: E402
from vision.app.run_live import LiveConfig  # noqa: E402
from vision.pipeline.batch_classifier import crop_with_context  # noqa: E402
from vision.pipeline.board_detector import DetectionResult, detect_board_on_frame  # noqa: E402

COLOR_DIRS = ("black", "empty", "white")
CSV_FIELDS = [
    "file", "split", "session", "shot_id", "timestamp", "fen", "raw_r", "raw_c", "square",
    "occ", "color", "symbol", "type_idx", "type_name", "context", "img_size_px",
    "exposure", "auto_exposure", "wb_temp", "auto_wb",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    live = LiveConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True, help="fényviszony/napszak azonosító, a fájlnevek prefixe (pl. delelott_ablak)")
    p.add_argument("--fen", default=None, help="kezdő FEN (ha nincs --fen-file)")
    p.add_argument("--fen-file", type=Path, default=None, help="soronként egy FEN; `#` komment; `n`-nel léptethető")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data_fen", help="kimeneti gyökér (alap: <repo>/data_fen)")
    p.add_argument("--burst", type=int, default=1, help="hány frame-et mentsen egy SPACE-re (kis időközzel)")
    p.add_argument("--burst-interval", type=float, default=0.25, help="másodperc a burst frame-ek között")
    p.add_argument("--split-mode", choices=("position", "shot", "train", "val"), default="position",
                   help="position: egy FEN minden fotója ugyanabba a splitbe (alap, nincs near-duplicate szivárgás); "
                        "shot: fotónként; train/val: minden ide")
    p.add_argument("--val-every", type=int, default=5, help="minden N. állás/fotó a val_new-ba (split-mode position/shot)")
    p.add_argument("--no-save-frames", action="store_true", help="ne mentse a teljes kameraképet")
    # kamera
    p.add_argument("--camera-index", type=int, default=live.camera_index)
    p.add_argument("--width", type=int, default=live.camera_width)
    p.add_argument("--height", type=int, default=live.camera_height)
    p.add_argument("--fps", type=int, default=live.camera_fps)
    p.add_argument("--exposure", type=float, default=None, help="MANUÁLIS exponálás (V4L2 exposure_time_absolute egysége); auto-exposure KI")
    p.add_argument("--wb-temp", type=float, default=None, help="fix fehéregyensúly (Kelvin); auto WB KI")
    p.add_argument("--drift-warn", type=float, default=2.5,
                   help="a tábla átlagfényességének szórása (0-255) az utolsó ~2 s-ban, ami felett exponálás-ingadozásra figyelmeztet")
    # címke-ellenőrzés a jelenlegi modellel
    p.add_argument("--weights", default=None, help="checkpoint az orientáció/FEN-ellenőrzéshez (alap: AppConfig.weights_path)")
    p.add_argument("--no-check-model", action="store_true", help="ne futtassa a modell-alapú egyezés-ellenőrzést")
    p.add_argument("--max-mismatch", type=int, default=14,
                   help="ennyi vagy több eltérő mező a FEN és a modell között -> a fotót NEM menti (--force felülírja)")
    p.add_argument("--force", action="store_true", help="mentés az ellenőrzés figyelmeztetései ellenére is")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Kamera
# ---------------------------------------------------------------------------

def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Nem sikerült megnyitni a kamerát (index={args.camera_index}).")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if args.exposure is not None:
        # V4L2: 1 = manual mode, 3 = aperture priority (auto)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
    if args.wb_temp is not None:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, args.wb_temp)
    return cap


def camera_state(cap: cv2.VideoCapture) -> dict[str, float]:
    return {
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        "wb_temp": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
        "auto_wb": cap.get(cv2.CAP_PROP_AUTO_WB),
        "gain": cap.get(cv2.CAP_PROP_GAIN),
    }


def print_exposure_banner(cap: cv2.VideoCapture, args: argparse.Namespace) -> None:
    st = camera_state(cap)
    auto_exp_on = st["auto_exposure"] not in (1.0,)          # V4L2: 1 = manual
    auto_wb_on = st["auto_wb"] not in (0.0,)
    print("\n" + "=" * 78)
    print("  KAMERA — exponálás / fehéregyensúly")
    print(f"  auto_exposure={st['auto_exposure']:.0f} ({'AUTO?' if auto_exp_on else 'manual'})  exposure={st['exposure']:.1f}  "
          f"auto_wb={st['auto_wb']:.0f} ({'AUTO?' if auto_wb_on else 'manual'})  wb_temp={st['wb_temp']:.0f}  gain={st['gain']:.0f}")
    if auto_exp_on or auto_wb_on:
        print("  !!! FIGYELEM: az auto-exposure / auto-WB valószínűleg BE van kapcsolva.")
        print("      A fekete bábu látszó fényereje így frame-ről frame-re változik — rögzítsd:")
        print(f"        python -m tools.collect_fen_dataset ... --exposure 250 --wb-temp 4600")
        print(f"      vagy:  v4l2-ctl -d /dev/video{args.camera_index} --set-ctrl=auto_exposure=1,exposure_time_absolute=250,"
              "white_balance_automatic=0,white_balance_temperature=4600")
        print("      (az értékeket a saját fényviszonyaidhoz állítsd; élesben UGYANEZT használd)")
    else:
        print("  OK: manuális exponálás és fehéregyensúly.")
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Tábla / kivágás — az éles útvonal
# ---------------------------------------------------------------------------

def detect(frame_bgr: np.ndarray, cfg: AppConfig) -> DetectionResult:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return detect_board_on_frame(gray, cell=cfg.cell, inner_pad_ratio=cfg.inner_pad_ratio)


def warp_and_crop(frame_bgr: np.ndarray, det: DetectionResult, cfg: AppConfig) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    """Ugyanaz, mint a ChessVisionTracker: warp -> crop_with_context(context=cfg.context)."""
    img_warp = cv2.warpPerspective(frame_bgr, det.M, cfg.warp_size, flags=cv2.WARP_INVERSE_MAP)
    rois = [[crop_with_context(img_warp, det.bbox_warp[r][c], context=cfg.context) for c in range(8)] for r in range(8)]
    return img_warp, rois


# ---------------------------------------------------------------------------
# Modell-alapú ellenőrzés (orientáció + FEN elgépelés ellen)
# ---------------------------------------------------------------------------

class ModelChecker:
    """
    A jelenlegi modellel megjósolja a 64 mező foglaltságát, és összeveti a
    FEN-ből származó nyers ráccsal. Két hibát fog meg:
      - a FEN nem azt írja le, ami a táblán van (elgépelés, rossz felállítás)
      - a tábla orientációja nem az, amit a pipeline feltételez — ilyenkor a
        FEN-rács valamelyik forgatott/tükrözött változata jobban egyezik.
    A modell a fekete bábun gyenge, ezért néhány eltérés normális; a küszöb
    (--max-mismatch) ehhez képest van lazán.
    """

    def __init__(self, weights_path: str):
        import torch
        from vision.models.occupancy_color_model import OccupancyColorModel
        from vision.pipeline.batch_classifier import preprocess_roi_for_batch

        self._torch = torch
        self._prep = preprocess_roi_for_batch
        self.model = OccupancyColorModel(weights_path=weights_path, device="cpu")

    def predict_occupancy(self, rois: list[list[np.ndarray]]) -> np.ndarray:
        torch = self._torch
        xs = [self._prep(rois[r][c], self.model) for r in range(8) for c in range(8)]
        with torch.no_grad():
            out = self.model.model(torch.stack(xs))
            logits = out[0] if isinstance(out, (tuple, list)) else out   # multi-task utód: (color, type)
            idx = logits.argmax(1).cpu().numpy()
        return self.model.idx_to_label[idx].reshape(8, 8).astype(np.int32)

    @staticmethod
    def symmetries(grid: np.ndarray) -> list[tuple[str, np.ndarray]]:
        out = []
        for k in range(4):
            g = np.rot90(grid, k)
            out.append((f"rot{90*k}", g))
            out.append((f"rot{90*k}+flip", np.fliplr(g)))
        return out

    def check(self, rois, fen_occ: np.ndarray) -> dict:
        pred = self.predict_occupancy(rois)
        mism = [(int(a), int(b)) for a, b in zip(*np.nonzero(pred != fen_occ))]
        best_name, best_n = "rot0", len(mism)
        for name, g in self.symmetries(fen_occ):
            n = int(np.count_nonzero(pred != g))
            if n < best_n:
                best_name, best_n = name, n
        return {"pred": pred, "mismatch": mism, "n_mismatch": len(mism), "best_symmetry": best_name, "best_n": best_n}


# ---------------------------------------------------------------------------
# Kimenet
# ---------------------------------------------------------------------------

class Writer:
    def __init__(self, out_dir: Path, session: str, save_frames: bool, cfg: AppConfig):
        self.out_dir, self.session, self.save_frames, self.cfg = out_dir, session, save_frames, cfg
        for split in ("train_new", "val_new"):
            for c in COLOR_DIRS:
                (out_dir / split / c).mkdir(parents=True, exist_ok=True)
        if save_frames:
            (out_dir / "frames").mkdir(parents=True, exist_ok=True)
        self.csv_path = out_dir / "labels.csv"
        new = not self.csv_path.exists()
        self._csv_f = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_f, fieldnames=CSV_FIELDS)
        if new:
            self._csv.writeheader()
        self._jsonl = open(out_dir / "shots.jsonl", "a", encoding="utf-8")
        self.n_shots = 0
        self.n_rois = 0
        self.per_class = {c: 0 for c in COLOR_DIRS}

    def close(self) -> None:
        self._csv_f.close()
        self._jsonl.close()

    def write_shot(self, *, frame, rois, labels: list[list[SquareLabel]], fen: str, split: str,
                   cam: dict, check: dict | None, shot_id: str) -> int:
        ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"
        base = f"{self.session}_{ts}"
        if self.save_frames:
            cv2.imwrite(str(self.out_dir / "frames" / f"{base}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved = 0
        for r in range(8):
            for c in range(8):
                lab = labels[r][c]
                roi = rois[r][c]
                if roi is None or roi.size == 0:
                    continue
                fname = f"{base}_r{r}c{c}_{lab.square}_{lab.symbol}.jpg"
                rel = Path(split) / lab.color_name / fname
                cv2.imwrite(str(self.out_dir / rel), roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self._csv.writerow({
                    "file": str(rel), "split": split, "session": self.session, "shot_id": shot_id, "timestamp": ts,
                    "fen": fen, "raw_r": r, "raw_c": c, "square": lab.square, "occ": lab.occ, "color": lab.color_name,
                    "symbol": lab.symbol, "type_idx": lab.type_idx, "type_name": lab.type_name,
                    "context": self.cfg.context, "img_size_px": roi.shape[0],
                    "exposure": cam.get("exposure"), "auto_exposure": cam.get("auto_exposure"),
                    "wb_temp": cam.get("wb_temp"), "auto_wb": cam.get("auto_wb"),
                })
                self.per_class[lab.color_name] += 1
                saved += 1
        self._jsonl.write(json.dumps({
            "shot_id": shot_id, "base": base, "session": self.session, "fen": fen, "split": split, "n_rois": saved,
            "camera": cam, "crop": {"context": self.cfg.context, "cell": self.cfg.cell, "inner_pad_ratio": self.cfg.inner_pad_ratio},
            "model_check": None if check is None else {
                "n_mismatch": check["n_mismatch"], "best_symmetry": check["best_symmetry"], "best_n": check["best_n"],
                "mismatch_rc": check["mismatch"]},
        }, ensure_ascii=False) + "\n")
        self._csv_f.flush(); self._jsonl.flush()
        self.n_shots += 1
        self.n_rois += saved
        return saved


def choose_split(mode: str, val_every: int, fen: str, shot_index: int) -> str:
    if mode == "train":
        return "train_new"
    if mode == "val":
        return "val_new"
    if mode == "shot":
        return "val_new" if (shot_index % max(1, val_every)) == val_every - 1 else "train_new"
    # position: az állás (FEN első 4 mezője) determinisztikus hash-e dönt
    key = " ".join(fen.split()[:4])
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return "val_new" if (h % max(1, val_every)) == 0 else "train_new"


# ---------------------------------------------------------------------------
# FEN kezelés
# ---------------------------------------------------------------------------

def validate_fen(fen: str) -> str | None:
    try:
        chess.Board(fen)
        return fen.strip()
    except ValueError as e:
        print(f"  ✗ Érvénytelen FEN: {e}")
        return None


def load_fen_file(path: Path) -> list[str]:
    fens = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if s and validate_fen(s):
            fens.append(s)
    return fens


def prompt_fen(current: str | None) -> str | None:
    print("\nÚj FEN (üres sor = marad a jelenlegi):")
    if current:
        print(f"  jelenlegi: {current}")
    try:
        s = input("  FEN> ").strip()
    except EOFError:
        return None
    if not s:
        return None
    return validate_fen(s)


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def draw_label_overlay(img_warp: np.ndarray, det: DetectionResult, labels, check: dict | None, size: int = 640) -> np.ndarray:
    vis = img_warp.copy()
    for r in range(8):
        for c in range(8):
            x0, y0, x1, y1 = det.bbox_warp[r][c]
            lab = labels[r][c]
            color = (0, 200, 0) if lab.occ == 1 else (0, 0, 255) if lab.occ == 2 else (160, 160, 160)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            txt = lab.square if lab.occ == 0 else f"{lab.symbol}{lab.square}"
            cv2.putText(vis, txt, (x0 + 4, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (x0 + 4, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            if check is not None and (r, c) in set(check["mismatch"]):
                cv2.rectangle(vis, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), (0, 255, 255), 3)
    h, w = vis.shape[:2]
    s = size / max(h, w)
    return cv2.resize(vis, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def put_text(img, text, y, color=(0, 255, 0)):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def board_mean_brightness(gray: np.ndarray, det: DetectionResult | None) -> float:
    if det is None or det.centers_img is None:
        return float(gray.mean())
    pts = np.asarray([det.centers_img[r][c] for r in range(8) for c in range(8)], dtype=np.float32)
    x0, y0 = np.clip(pts.min(0), 0, [gray.shape[1] - 1, gray.shape[0] - 1]).astype(int)
    x1, y1 = np.clip(pts.max(0), 0, [gray.shape[1], gray.shape[0]]).astype(int)
    if x1 <= x0 or y1 <= y0:
        return float(gray.mean())
    return float(gray[y0:y1, x0:x1].mean())


# ---------------------------------------------------------------------------
# Fő ciklus
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = AppConfig()

    fen_list: list[str] = load_fen_file(args.fen_file) if args.fen_file else []
    fen_idx = 0
    fen: str | None = None
    if fen_list:
        fen = fen_list[0]
    elif args.fen:
        fen = validate_fen(args.fen)
    if fen is None:
        fen = prompt_fen(None)
    if fen is None:
        raise SystemExit("Nincs érvényes FEN — add meg --fen vagy --fen-file kapcsolóval.")

    checker: ModelChecker | None = None
    if not args.no_check_model:
        wp = args.weights or cfg.weights_path
        if wp and Path(wp).exists():
            print(f"Ellenőrző modell betöltése: {wp}")
            try:
                checker = ModelChecker(wp)
            except Exception as e:  # a gyűjtés a modell nélkül is menjen
                print(f"  (modell-ellenőrzés kikapcsolva, betöltési hiba: {e})")
        else:
            print("  (nincs checkpoint, a modell-alapú FEN/orientáció-ellenőrzés kimarad)")

    writer = Writer(args.out_dir, args.session, save_frames=not args.no_save_frames, cfg=cfg)
    cap = open_camera(args)
    print_exposure_banner(cap, args)

    print(f"Session: {args.session} | kimenet: {args.out_dir.resolve()}")
    print(f"FEN [{fen_idx + 1}/{max(1, len(fen_list))}]: {fen}")
    print("SPACE=fotó  n=következő FEN  f=FEN bekérés  d=tábla újradetektálás  o=overlay  e=exponálás  q=kilépés")

    det: DetectionResult | None = None
    labels = fen_to_raw_labels(fen)
    fen_occ = fen_to_raw_occupancy(fen)
    show_overlay = True
    last_check: dict | None = None
    brightness = deque(maxlen=60)
    shot_index = 0
    status = "tábla: nincs detektálva (d vagy SPACE)"

    def redetect(frame) -> bool:
        nonlocal det, status
        t0 = time.perf_counter()
        d = detect(frame, cfg)
        dt = (time.perf_counter() - t0) * 1000
        if not d.ok:
            status = f"tábla: detektálás SIKERTELEN ({dt:.0f} ms)"
            print(f"  ✗ {status}")
            return False
        det = d
        status = f"tábla: OK ({dt:.0f} ms)"
        print(f"  ✓ tábla detektálva ({dt:.0f} ms)")
        return True

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(board_mean_brightness(gray, det))
            drift = float(np.std(brightness)) if len(brightness) >= 15 else 0.0

            preview = frame.copy()
            split = choose_split(args.split_mode, args.val_every, fen, shot_index)
            put_text(preview, f"[{args.session}] fotók: {writer.n_shots}  ROI: {writer.n_rois}  "
                             f"(b={writer.per_class['black']} e={writer.per_class['empty']} w={writer.per_class['white']})  -> {split}", 28)
            put_text(preview, f"FEN {fen_idx + 1}/{max(1, len(fen_list))}: {fen[:60]}", 54)
            put_text(preview, status, 80, (0, 255, 0) if det is not None else (0, 165, 255))
            if drift > args.drift_warn:
                put_text(preview, f"!!! FÉNYERŐ INGADOZIK (std={drift:.1f}) — auto-exposure be van kapcsolva?", 106, (0, 0, 255))
            if last_check is not None:
                col = (0, 255, 0) if last_check["n_mismatch"] < 6 else (0, 165, 255)
                put_text(preview, f"modell vs FEN: {last_check['n_mismatch']} eltérő mező (legjobb szimmetria: {last_check['best_symmetry']}, {last_check['best_n']})", 132, col)
            if det is not None and det.centers_img is not None:
                for r in range(8):
                    for c in range(8):
                        x, y = det.centers_img[r][c]
                        cv2.circle(preview, (int(x), int(y)), 3, (0, 255, 255), -1)
            cv2.imshow("FEN-gyűjtő — kamera", preview)

            if show_overlay and det is not None:
                img_warp = cv2.warpPerspective(frame, det.M, cfg.warp_size, flags=cv2.WARP_INVERSE_MAP)
                cv2.imshow("FEN címkék a warpolt táblán (zöld=fehér, piros=fekete, sárga=eltérés a modelltől)",
                           draw_label_overlay(img_warp, det, labels, last_check))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                redetect(frame)
            elif key == ord("o"):
                show_overlay = not show_overlay
                if not show_overlay:
                    cv2.destroyWindow("FEN címkék a warpolt táblán (zöld=fehér, piros=fekete, sárga=eltérés a modelltől)")
            elif key == ord("e"):
                print_exposure_banner(cap, args)
            elif key in (ord("n"), ord("f")):
                new_fen = None
                if key == ord("n") and fen_list:
                    fen_idx = (fen_idx + 1) % len(fen_list)
                    new_fen = fen_list[fen_idx]
                else:
                    new_fen = prompt_fen(fen)
                if new_fen:
                    fen = new_fen
                    labels = fen_to_raw_labels(fen)
                    fen_occ = fen_to_raw_occupancy(fen)
                    last_check = None
                    print(f"FEN [{fen_idx + 1}/{max(1, len(fen_list))}]: {fen}  -> split: {choose_split(args.split_mode, args.val_every, fen, shot_index)}")
                    print("  Állítsd fel a táblát, ellenőrizd az overlay-t, majd SPACE.")
            elif key == ord(" "):
                if det is None and not redetect(frame):
                    continue
                if drift > args.drift_warn and not args.force:
                    print(f"  ! Fényerő-ingadozás (std={drift:.1f} > {args.drift_warn}) — rögzítsd az exponálást, vagy --force.")
                    continue
                shot_id = f"{args.session}_{int(time.time() * 1000)}"
                saved_total = 0
                for b in range(max(1, args.burst)):
                    if b > 0:
                        time.sleep(args.burst_interval)
                        ok, frame = cap.read()
                        if not ok:
                            continue
                    img_warp, rois = warp_and_crop(frame, det, cfg)
                    check = checker.check(rois, fen_occ) if checker is not None else None
                    if check is not None:
                        last_check = check
                        if check["best_symmetry"] != "rot0" and check["n_mismatch"] - check["best_n"] >= 6:
                            print(f"  !!! ORIENTÁCIÓ-GYANÚ: a FEN-rács '{check['best_symmetry']}' változata {check['best_n']} eltérést ad, "
                                  f"az alap {check['n_mismatch']}-t. Ellenőrizd a tábla állását / a FEN-t az overlay-en.")
                            if not args.force:
                                print("      NEM mentem (--force felülírja).")
                                break
                        if check["n_mismatch"] >= args.max_mismatch:
                            sq = ", ".join(labels[r][c].square for r, c in check["mismatch"][:12])
                            print(f"  !!! {check['n_mismatch']} mező eltér a modell szerint ({sq}...) — rossz FEN vagy felállítás?")
                            if not args.force:
                                print("      NEM mentem (--force felülírja).")
                                break
                    saved = writer.write_shot(frame=frame, rois=rois, labels=labels, fen=fen, split=split,
                                              cam=camera_state(cap), check=check, shot_id=shot_id)
                    saved_total += saved
                if saved_total:
                    shot_index += 1
                    msg = f"  ✓ {saved_total} ROI mentve -> {split}  | össz: {writer.n_shots} fotó, {writer.n_rois} ROI"
                    if last_check is not None:
                        msg += f" | modell-eltérés: {last_check['n_mismatch']}"
                    print(msg)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        writer.close()

    print(f"\nKész. {writer.n_shots} fotó, {writer.n_rois} ROI -> {args.out_dir.resolve()}")
    print(f"  osztályonként: {writer.per_class}")
    print("  Tanítás: python tools/train_square_classifier.py --data-root <github dataset> "
          f"--data-root {args.out_dir} --eval-dir {args.out_dir}")


if __name__ == "__main__":
    main()
