# Rendszerleírás — technikai dokumentáció

## Rendszerarchitektúra áttekintés

A rendszer négy rétegből áll, amelyek HTTP-n és WebSocket-en kommunikálnak egymással:

```
[Kamera] → [Vision pipeline] → [Backend (FastAPI:8001)] → [Robot réteg] → [Franka container (HTTP:8002)]
                                       ↑
                              [Frontend (WebSocket)]
```

| Réteg | Fő fájl(ok) | Port | Protokoll |
|---|---|---|---|
| Vision | `vision/app/run_live.py` | — | HTTP POST a backendhez |
| Backend | `backend/main.py` | 8001 | REST + WebSocket |
| Robot | `robot/impl/franka.py` | — | HTTP kliens |
| Franka executor | `franka/chess_executor.py` | 8002 | REST (ROS2-ból) |
| Frontend | `frontend/` | 8000 | WebSocket kliens |

---

## Vision pipeline

### Kamerabeolvasás (`vision/app/run_live.py`)

A `LiveConfig` dataclass konfigurálja a kameraforrást:

- Felbontás: 1280×720, 30 FPS (`cv2.VideoCapture`)
- Minden hatodik képkockát (`process_every_nth_captured_frame = 6`) dolgoz fel a rendszer
- A `BackendSyncClient` HTTP POST-tal küldi az eredményeket a backend `/api/move` endpointjára

### Tábladetektálás (`vision/pipeline/board_detector.py`, `vision/board/find_chessboard.py`)

A `detect_board_on_frame(gray, *, cell=96, inner_pad_ratio=0.06)` függvény `DetectionResult`-ot ad vissza (mezők: `ok`, `M`, `bbox_warp`, `centers_warp`, `centers_img`).

**Algoritmus:**

1. **Nyeregpont-detektálás** (`getSaddle(gray_img)`): Hessian-determináns kiszámítása pixel-szinten (gxx·gyy − gxy²). A sakktábla sarokpontjain a determináns negatív, ami megkülönbözteti őket más struktúráktól.
2. **Non-maximal suppression** (`nonmax_sup(img, win=10)`): 10×10 ablakban csak a lokális maximum marad.
3. **Kontúrvalidálás** (`is_square(cnt, eps=3.0)`): A négyzetes kontúrokat szűri oldalarány, átló-arány (min/max > 0.5) és szögek (40°–140°) alapján.
4. **Homográfia** (`generateNewBestFit`): A talált belső rácspontokból RANSAC-alapú homográfiamátrix számítás, amely a ferde kameraképet top-down nézetbe transzformálja.
5. **Warp**: A kép átméretezése 1632×1632 pixelre (17×96), majd a 64 mező bounding boxainak kiszámítása (`extract_squares_from_warp`), belső padding: `inner_pad_ratio=0.06`.

A `ChessVisionTracker`-ben opcionális Lucas-Kanade optikai áramlás-alapú követés is szerepel: ha a tábla már meg van találva, a nyeregpontokat a következő képkockán követi, és csak periodikusan végez teljes újradetektálást.

### Képosztályozás (`vision/pipeline/batch_classifier.py`)

**Modell:** `OccupancyColorModel` (ResNet18, 3 kimeneti osztály: `empty=0`, `white=1`, `black=2`).

- Súlyfájl: `vision/models/weights/resnet18_best_topdown.pt`
- Bemeneti méret: 100×100 pixel
- Normalizálás: mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]
- Kontextus: minden mező körül 50%-os padding (`context=0.50`)

**Batch osztályozás** (`classify_frame_batch`): A warped képből az összes 64 mezőt egyszerre osztályozza — egy batch forward pass-szal.

**Partial reclassify optimalizáció** (`classify_selected_squares`):
- `compute_square_diffs()`: pixeldifferenciát számít az előző képkockához képest
- Csak azok a mezők kerülnek újra a hálóba, ahol a diff > `partial_diff_threshold=18.0`
- Maximum `partial_max_squares=12` mező/képkocka
- Minden 30. képkocka teljes osztályozás (`full_reclassify_interval=30`)

**Koordináta-transzformáció** (`raw_to_standard`): A kamera perspektívájából 90° CW forgatás + vízszintes tükrözés → standard sakktábla-orientáció (bal felső: A8, jobb alsó: H1).

### StateStabilizer (`vision/pipeline/stabilizer.py`)

A nyers osztályozó kimenet zajos; a `StateStabilizer` szavazásos szűrést alkalmaz.

**Konstruktor paraméterei (AppConfig szerint):**

| Paraméter | Érték | Jelentés |
|---|---|---|
| `buffer_size` | 5 | Utolsó N képkocka tárolása |
| `min_votes_ratio` | 0.65 | Szavazási küszöb (65% egyező kell) |
| `stable_frames` | 4 | Ennyiszer kell látni egymás után |
| `emit_cooldown_s` | 0.20 s | Két emisszió között minimális idő |
| `min_mean_conf` | 0.50 | Átlagos bizalom küszöb |
| `min_changed_conf` | 0.50 | Változott mezők minimális bijalma |
| `max_changed_for_move` | 6 | Egy lépésnél max. ennyi mező változhat |
| `hold_changed_threshold` | 10 | Ennyi felett HOLD mód |
| `hold_low_conf_threshold` | 0.35 | Alacsony bizalom → HOLD |
| `hold_min_duration_s` | 0.50 s | HOLD minimális időtartama |

**Állapotgép:**

```
WARMUP → CANDIDATE → STABLE
              ↕
            HOLD
```

- `WARMUP`: Buffer feltöltése (< 3 képkocka)
- `CANDIDATE`: Bázisállapot meghatározása folyamatban
- `STABLE`: Baseline megvan, lépések felismerhetők
- `HOLD`: Nagy perturbáció (kéz a táblán stb.) — `recovery_stable_frames=2` megerősítő frame szükséges a kilépéshez

**`update(occ, confs)` logika:**
1. Cell-wise majority vote a bufferen → kandidátus occupancy + vote strength
2. Ha a kandidátus megegyezik az utolsó emittált állással, növeli a run counter-t
3. Validáció: vote_strength ≥ 0.65, mean_conf ≥ 0.50
4. Lépésnél: hamming_distance ≤ 6, changed_conf ≥ 0.50
5. emit_cooldown betartása
6. Hamming > 10 → HOLD mód

---

## Chess logic réteg

### Move resolver (`chess_logic/resolver.py`)

A `resolve_move_from_occupancy(current_board, observed_occ, observed_conf, *, max_noise_cells=1, max_weighted_cost=0.9)` a látott occupancy-ból azonosítja a lépést.

**Exact matching:**
Végigiterál a python-chess `board.legal_moves`-on. Minden lépésre kiszámolja az elvárt occupancy-t (`board_to_occupancy`), és összehasonlítja a megfigyelttel. Ha nulla eltérés → exact match, kész.

**Fuzzy matching** (ha az exact nem sikerül):
- `occupancy_distance(a, b)`: Hamming-távolság (különböző cellák száma)
- Szűrő: distance ≤ `max_noise_cells=1`
- `weighted_diff(observed_occ, expected_occ, conf)`: a különböző cellákon lévő konfidenciák összege
- Szűrő: weighted_cost ≤ `max_weighted_cost=0.9`
- A legkisebb cost-ú legális lépés kerül kiválasztásra

**Visszatérési érték:** `(MoveGuess | None, expected_occ, mode)` ahol mode: `"exact"` / `"fuzzy c=X n=Y"` / `None`

### Game state (`chess_logic/game.py`)

A `Game` osztály a python-chess `Board`-ot wrappolja:
- `apply_uci(uci: str)` → `MoveGuess | None` (szinkron lépésalkalmazás)
- `resolve_from_occupancy(...)` → `OccupancyResolveResult`
- `to_pgn(...)` → PGN string export
- `move_history`: `MoveRecord` lista (FEN before/after, SAN, check/checkmate/stalemate flag-ek, döntetlen feltételek)

### Move típusok (`chess_logic/move_types.py`)

A `MoveGuess` dataclass tartalmazza a lépés összes jellemzőjét:

| Mező | Típus | Leírás |
|---|---|---|
| `from_row, from_col` | int | Forrás koordináta (vision rendszerben) |
| `to_row, to_col` | int | Cél koordináta |
| `piece` | str | Bábu szimbólum ("P", "p", "N" stb.) |
| `captured` | bool | Ütés-e |
| `is_castling` | bool | Sáncolás |
| `castling_side` | str\|None | "K" (rövid) vagy "Q" (hosszú) |
| `is_en_passant` | bool | En passant |
| `promotion_piece` | str\|None | "q", "r", "b", "n" |

---

## Backend réteg (`backend/`)

### FastAPI app (`backend/main.py`)

CORS middleware: localhost:8000, :3000, :5173 engedélyezve.

### Endpointok (`backend/api/routes.py`)

| Metódus | Útvonal | Leírás |
|---|---|---|
| GET | `/api/health` | `{"ok": true}` |
| GET | `/api/state` | Teljes játékállapot |
| POST | `/api/new-game?fen=...` | Új játék indítása |
| POST | `/api/move` | UCI lépés alkalmazása (body: `{"uci": "e2e4"}`) |
| GET | `/api/robot/busy` | `{"busy": bool}` |
| GET | `/api/robot/best-move` | Stockfish legjobb lépés |
| WebSocket | `/api/ws/state` | Valós idejű állapotfrissítés |

### BackendState (`backend/core/state.py`)

Thread-safe (threading.Lock) central state:
- `game: Game` — aktuális játékállás
- `robot_service: RobotExecutionService | None`
- `engine_service: EngineAnalysisService`
- `robot_busy: bool`

**Robot auto-play logika:** Az elemzés befejeztekor, ha a robot következik és a kívánt szín (`robot_color="black"`) az ő köre, a `BackendState` háttérszálban meghívja `_run_robot_move(uci, board)` — ez alkalmazás a lépést, frissíti az állapotot, és kiszórja WebSocket-en.

### Stockfish integráció (`backend/services/engine_service.py`)

`EngineAnalysisService(engine_path="/usr/games/stockfish", default_depth=16, default_multipv=3)`

- UCI protokollon át kommunikál (`chess.engine.SimpleEngine.popen_uci`)
- Háttérszálas elemzési sor (queue): `schedule_analysis(snapshot, on_ready)` callback
- Kimenet (per sor): `score` (centipawn), `line_san` (emberi jelölés), `pv_uci` (lista), `mate_in` (ha matt)

### WebSocket hub (`backend/services/ws_service.py`)

`WebSocketHub.broadcast_state(state)` — minden csatlakozott kliensnek elküldi az aktuális állapot JSON-ját. Thread-safe wrapper: `notify_state_changed()`.

---

## Robot réteg (`robot/`)

### Kalibráció (`robot/calibration.py`)

A `Calibration` osztály az A1 és H8 sarokpontból interpolálja mind a 64 mező pozícióját robot-koordináta rendszerben (mm).

**Matematika:**
```
dx, dy = H8 - A1
file_x = (dx + dy) / 14.0
file_y = (dy - dx) / 14.0
rank_step = (-file_y, file_x)   # 90° CCW forgatás
```

`square_to_xy("e4")` → az adott mező középpontjának (x, y) koordinátái mm-ben.

Mentés/betöltés: `calibration.json` (tartalmazza: `a1`, `h8`, `graveyard_start`, `graveyard_step`, `promotion_queen_xy`).

### Graveyard (`robot/graveyard.py`)

A leütött bábukat tároló terület: `start=(400.0, 0.0)`, lépésköz `step=(30.0, 0.0)` mm. Maximum 15 slot (`MAX_GRAVEYARD_SLOTS`). A `next_slot()` mindig a következő szabad pozíciót adja vissza és inkrementálja a számlálót. Új játéknál `reset()`.

### Move descriptor (`robot/move_descriptor.py`)

A `build_move_descriptor(uci, board, calibration, graveyard)` függvény egy dict-et épít:

| `type` | Tartalom |
|---|---|
| `"simple"` | `piece_from_xy`, `piece_to_xy` |
| `"capture"` | + `captured_xy`, `graveyard_xy` |
| `"en_passant"` | Speciális: az ütött gyalog nem a célmezőn van |
| `"castling"` | + `castling_rook: {rook_from_xy, rook_to_xy}` |
| `"promotion"` | + `pawn_graveyard_xy`, `promotion_target_xy`, `promotion` |

---

## Franka container (`franka/chess_executor.py`)

### MoveIt2 konfiguráció

- Base frame: `fr3_link0`
- End-effector: `fr3_hand_tcp`
- Group: `fr3_arm` (fr3_joint1..7)
- Gripper lefelé mutató orientáció: `_DOWN_QUAT = [1.0, 0.0, 0.0, 0.0]`
- Max sebesség: 0.3 m/s, gyorsulás: 0.3 m/s²

### Fizikai paraméterek

| Konstans | Érték | Szerepe |
|---|---|---|
| `Z_LIFT` | 0.12 m | Tábla feletti biztonságos magasság |
| `Z_TRAVEL` | 0.20 m | Ív tetőpontja |
| `Z_PICK` | 0.04 m | Megközelítési magasság felvételhez |
| `Z_PLACE` | 0.03 m | Megközelítési magasság letételhez |
| `ARC_WAYPOINTS` | 8 | Közbülső pontok száma az ívben |
| `GRIPPER_OPEN_WIDTH` | 0.06 m | Nyitott fogó szélessége |
| `GRIPPER_GRASP_FORCE` | 20.0 N | Fogóerő |
| `GRIPPER_GRASP_SPEED` | 0.05 m/s | Fogás sebessége |
| `GRIPPER_GRASP_EPS` | 0.015 m | Pozíció-tűrés fogásnál |

### Mozgástervezés

**Vertikális Cartesian path** (`_vertical_path`): Fix XY pozíción fel-le mozgás, MoveIt2 Cartesian path plannerrel.

**Ív** (`_arc_path`): Parabolikus Z-profil 8 waypoint-on:
```
z(t) = Z_LIFT + 4·(Z_TRAVEL − Z_LIFT)·t·(1−t)
```
t=0 és t=1: Z_LIFT; t=0.5: Z_TRAVEL.

### Pick-and-place szekvencia (`_pick_and_place`)

1. Leereszkedés Z_PICK-re → fogás (20 N)
2. Emelkedés Z_LIFT-re
3. Ívelt átvitel a célmező fölé
4. Leereszkedés Z_PLACE-re → fogó kinyitása
5. Emelkedés Z_LIFT-re

### HTTP API (port 8002)

| Metódus | Útvonal | Leírás |
|---|---|---|
| GET | `/health` | `{"ok": true}` |
| GET | `/position` | `{"x", "y", "z"}` mm-ben |
| POST | `/execute` | Move descriptor JSON végrehajtása |
| POST | `/gripper/open` | Fogó kinyitása |
| POST | `/gripper/close` | Fogás (grasp) |
| POST | `/move` | Direkt pozíció (kalibrációhoz) |

---

## Konfigurációs fájlok és környezeti változók

| Fájl / Env var | Alapértelmezett | Leírás |
|---|---|---|
| `robot/calibration.json` | — | Kötelező (a1, h8, graveyard, promotion_queen_xy) |
| `vision/models/weights/resnet18_best_topdown.pt` | — | ResNet18 súlyok |
| `BACKEND_URL` | `http://127.0.0.1:8001` | Vision → Backend URL |
| `Settings.stockfish_path` | `/usr/games/stockfish` | Stockfish bináris |
| `Settings.robot_color` | `"black"` | Melyik oldalt játssza a robot |
| `Settings.deep_depth` | 16 | Elemzési mélység |
| `Settings.deep_multipv` | 3 | Top N sor |
| `Settings.robot_enabled` | `True` | Robot bekapcsolva |

---

## Indítási sorrend

1. **Franka container** (`run.sh real <robot_ip>` vagy `run.sh sim`):
   - Docker container elindítása (`franka_ros2_humble`)
   - `chess_executor.py` bemásolása a containerbe
   - Tmux session ("chess") nyitása 2 ablakkal:
     - Felső: `ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=... use_fake_hardware:=...`
     - Alsó: 8 mp várakozás, majd `python3 /ros2_ws/chess_executor.py` (port 8002)

2. **Backend**:
   ```
   python3 -m backend.main   # vagy: uvicorn backend.main:app --port 8001
   ```
   - Betölti a `calibration.json`-t
   - Inicializálja a Stockfish engine-t
   - Csatlakozik a Franka executorhoz (port 8002, max 20 újrapróbálkozás)

3. **Vision pipeline**:
   ```
   python3 -m vision.app.run_live
   ```
   - Betölti a ResNet18 súlyokat
   - Kamera megnyitása (index 0)
   - Reseteli a backendet (`/api/new-game`)
   - Inicializáló fázis: tábladetektálás, 3 frame-es baseline felvétel
   - Folyamatos loop: képkocka → detektálás → osztályozás → stabilizálás → resolver → backend POST

4. **Frontend**: Böngészőben megnyitva (port 8000), WebSocket csatlakozás a backendhez.
