from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vision.pipeline.stabilizer import StabilizerParams, StateStabilizer


DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "models" / "weights" / "mnv3_squares_128.pt"


@dataclass
class AppConfig:
    start_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    weights_path: str | None = str(DEFAULT_WEIGHTS_PATH)

    # Inferencia backend: "onnx" (alapértelmezett, ~8x gyorsabb CPU-n), "torch"
    # (fallback), "auto" (onnx ha van .onnx a .pt mellett, különben torch).
    # Az ONNX fájl: python -m tools.export_onnx --weights <weights_path>
    # (alapból <weights_path>.onnx-ra ír; onnx_path=None ezt keresi).
    inference_backend: str = "auto"
    onnx_path: str | None = None
    inference_threads: int | None = None
    # Ha az export (--int8) egy INT8 változatot ajánlott (black recall nem
    # romlott), auto módban azt töltjük; False -> mindig a fp32 .onnx.
    allow_int8: bool = True

    inner_pad_ratio: float = 0.06
    context: float = 0.50
    cell: int = 96

    init_buffer_frames: int = 3
    init_max_dist: int = 2

    # Resolver zajtűrése (chess_logic.resolver): legfeljebb ennyi mezőben
    # térhet el a megfigyelt rács egy legális lépés elvárt rácsától (fuzzy),
    # és a zajos mezők konfidencia-összege legfeljebb ennyi lehet. A fuzzy
    # találathoz a célmezőt foglaltnak kell látni, és a megfigyelt rácsnak
    # >= 2 mezőben kell eltérnie az elfogadott állástól (invariáns).
    fuzzy_max_noise_cells: int = 1
    fuzzy_max_weighted_cost: float = 0.9

    # Bábutípus-fej: a promóciós bábu kiválasztásához (csak ott!). Ha a
    # célmezőn a legvalószínűbb promóciós típus valószínűsége ez alatt van,
    # vezér az alapértelmezés. Ha a típus-fej a célmezőn még GYALOGOT lát
    # (a játékos még nem cserélte le), legfeljebb promotion_wait_s-ig várunk
    # a cserére, utána a resolver választása szerint fogadjuk el.
    promotion_min_conf: float = 0.50
    promotion_wait_s: float = 5.0
    # Kapcsoló a jövőbeli kiterjesztéshez: a típus-fej a teljes
    # lépésdetektálásban (azonos költségű fuzzy jelöltek között dönt).
    # ALAPBÓL KI — a 3-osztályos út érintetlen.
    use_type_hint_for_moves: bool = False

    enable_pipeline_profiler: bool = True

    # --- képkockánkénti klasszifikáció -------------------------------------
    partial_reclassify: bool = True
    # Négyzetenkénti mean|diff| (0-255) az előző feldolgozott frame-hez képest,
    # ami felett a mezőt újraklasszifikáljuk. Mért statikus alapvonal: max 7,1.
    partial_diff_threshold: float = 18.0
    # A hand over the board dirties well over 12 squares; the ones that don't
    # fit kept stale labels and poisoned the vote. ~0.35 ms/square (ORT,
    # MobileNetV3 @128), so 20 is far inside the per-frame budget.
    partial_max_squares: int = 20
    # Gördülő frissítés: minden frame-ben ennyi további mezőt osztályozunk
    # újra körbejárva (8 -> a 64 mező ~8 frame = ~270 ms alatt frissül 30
    # fps-en). Ez váltja ki a régi "rossz címke megmarad a következő teljes
    # újraklasszifikálásig" problémát (10. szakasz 3. pont) a teljes út
    # költségének töredékéért (28 ROI: ~11 ms vs 64 ROI: ~30 ms).
    rolling_refresh_squares: int = 8
    # Teljes újraklasszifikálás ennyi frame-enként (biztonsági háló; 30 fps-en
    # 3 s). A gördülő frissítés miatt ritkább lehet, mint a régi 45.
    full_reclassify_interval: int = 90
    # Eseményvezérelt teljes újraklasszifikálás: ha a stabilizer tartós, de
    # nem feloldható eltérést lát (1 mezős eltérés, vagy kiadott rács legális
    # lépés nélkül), legfeljebb ennyi időnként teljes átosztályozás.
    wakeup_full_reclassify_s: float = 0.5

    # --- mozgás-kapu (a stabilizer motion bemenete) ------------------------
    # A mozgást az előző feldolgozott frame-hez ÉS egy ~motion_ref_age_s-mal
    # korábbi frame-hez képest is mérjük (30 fps-en két szomszédos frame
    # között egy lassan mozgó kéz diffje kicsi lehet); a kettő maximuma megy
    # a stabilizernek. A küszöb: StabilizerParams.motion_threshold.
    motion_ref_age_s: float = 0.10

    # --- elfogadási politika a trackerben ----------------------------------
    # Ha a feloldott lépés egy másik legális lépés "előtagja" (a bástyával
    # kezdett sánc Rf1-nek látszik, miközben O-O készül), ennyivel hosszabb
    # stabilitást várunk, mielőtt a rövidebb lépést elfogadjuk.
    prefix_ambiguity_extra_s: float = 1.0

    # --- tábla-újradetektálás ----------------------------------------------
    # The homography is solved once at init and then frozen, so a nudged board
    # or camera degrades every later classification with no way back. When the
    # stabilizer cannot settle for this long, re-solve it (~165 ms 1080p-n a
    # vektorizált detektorral; régen 1,4-2,2 s).
    redetect_after_stuck_s: float = 4.0
    redetect_min_interval_s: float = 5.0
    # Újradetektálás után a rács orientációja NEM garantált (mérve: a mentett
    # képek 18/184-ét 270°-kal forgatva adta vissza a detektor). Az elfogadott
    # álláshoz képest a 8 szimmetria közül azt választjuk, amelyik EGYÉRTELMŰEN
    # illeszkedik (legfeljebb align_max_mismatch eltérés ÉS legalább
    # align_min_margin mezővel jobb a második legjobbnál), és a bbox-rácsot
    # ahhoz rendezzük át. A raw_to_standard leképezés érintetlen.
    redetect_align_orientation: bool = True
    align_max_mismatch: int = 2
    align_min_margin: int = 8

    @property
    def warp_size(self):
        s = 17 * self.cell
        return (s, s)


# A stabilizer küszöbei — EGY helyen. Az egyes mezők mérési indoklása a
# StabilizerParams definíciójában (vision/pipeline/stabilizer.py); a mérések:
# docs/pipeline_tuning.md, tools/measure_pipeline_noise.py, tools/replay_frames.py.
STABILIZER_PARAMS = StabilizerParams(
    min_stable_s=0.25,
    min_stable_frames=3,
    motion_threshold=12.0,
    min_static_s=0.20,
    flicker_tolerance_cells=1,
    max_candidate_misses=3,
    min_changed_for_move=2,
    max_changed_for_move=6,
    min_changed_conf=0.40,
    frame_reject_mean_conf=0.60,
)


def make_stabilizer() -> StateStabilizer:
    return StateStabilizer(STABILIZER_PARAMS)
