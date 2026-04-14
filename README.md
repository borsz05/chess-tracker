# Chess Vision

A real-time chess game tracking system that uses a camera and a convolutional neural network to detect board state, recognize moves, analyze positions with Stockfish, and display live analysis in a browser. An optional robot arm layer can physically execute moves on the board.

---

## System Architecture

The project is organized into five layers. Data flows from left to right:

```
Camera → [vision] → [chess_logic] → [backend] → [frontend]
                                         ↑
                                      [robot]  (optional)
```

| Layer | Role |
|---|---|
| **chess_logic** | Game state, move history, occupancy-to-move resolution, stabilizer |
| **vision** | Board detection (saddle points + homography), square crop, ResNet18 classification, partial reclassification, live processing loop |
| **backend** | FastAPI REST + WebSocket server, Stockfish analysis (background thread), state broadcast |
| **robot** | Abstract interface + calibration for physical robot arm (optional) |
| **frontend** | Browser UI — live chessboard, engine lines, move list, eval bar |

**Data flow detail:**

1. `LatestFrameCamera` captures frames in a background thread (buffer size 1 — always latest).
2. `LiveProcessor` processes every Nth frame (default: every 6th) in a second background thread.
3. `ChessVisionTracker` detects the board once (initialization), then warps each frame, classifies 64 squares with a ResNet18 batch inference, and stabilizes the result with `StateStabilizer`.
4. When a stable new board position is detected, `resolve_move_from_occupancy` identifies the legal move that matches the observed occupancy.
5. The accepted move is pushed to the backend via `BackendSyncClient` (HTTP POST).
6. The backend stores the move in its own `Game` instance, schedules a Stockfish analysis, and broadcasts the updated state to all WebSocket clients.
7. The frontend receives the state push and selectively re-renders only the changed UI components.

---

## Requirements

- **Python** 3.11+
- **Key Python packages** (see `requirements.txt`):
  - `opencv-python` >= 4.10 — camera capture and image processing
  - `torch` >= 2.5, `torchvision` >= 0.20 — ResNet18 classifier
  - `python-chess` >= 1.999 — move generation and validation
  - `fastapi` >= 0.115, `uvicorn[standard]` >= 0.34 — backend server
  - `pydantic` >= 2.10 — request/response models
  - `numpy` >= 1.26
- **Stockfish** — external binary, path configured in `backend/core/config.py`
- **Frontend** — static HTML/JS/CSS, no build step required; open directly in any modern browser

---

## How to Run

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8001
```

The backend will be available at `http://localhost:8001`. On startup it initializes the game state and immediately queues a Stockfish analysis of the starting position.

### 3. Open the frontend

Open `frontend/index.html` directly in a browser (no server needed). It connects to the backend at `http://localhost:8001` and `ws://localhost:8001/api/ws/state`.

You can also serve it with any static server:
```bash
python -m http.server 8000 --directory frontend
```
Then open `http://localhost:8000`.

### 4. Start the vision pipeline

In a separate terminal, from the project root:

```bash
python -m vision.app.run_live
```

A preview window opens showing the camera feed with a status overlay. The pipeline detects the board, waits for it to reach the starting position, then begins tracking moves and pushing them to the backend.

**Keyboard shortcuts in the preview window:**
- `q` — quit
- `r` — reset tracker (and optionally sync a new game to the backend)

---

## Configuration

### Vision pipeline — `vision/app/config.py`

`AppConfig` is a dataclass with all tunable vision parameters. The most important fields:

| Field | Default | Description |
|---|---|---|
| `weights_path` | `vision/models/weights/resnet18_best_topdown.pt` | Path to the trained classifier checkpoint |
| `cell` | 96 | Pixel size of each square in the warped image |
| `context` | 0.50 | Padding around each square crop fed to the model |
| `inner_pad_ratio` | 0.06 | Fraction of each square's border trimmed before cropping |
| `init_buffer_frames` | 3 | Frames averaged for the initial position baseline |
| `init_max_dist` | 2 | Max Hamming distance from the expected start position during init |
| `fuzzy_max_noise_cells` | 1 | Max cells that may differ in fuzzy move resolution |
| `fuzzy_max_weighted_cost` | 0.9 | Max confidence-weighted difference in fuzzy resolution |
| `partial_reclassify` | True | Only re-classify squares that changed (faster) |
| `full_reclassify_interval` | 30 | Force a full classification every N frames |

### Live processing — `vision/app/run_live.py`

`LiveConfig` controls camera settings and runtime behavior:

| Field | Default | Description |
|---|---|---|
| `camera_index` | 0 | OpenCV camera index |
| `camera_width` / `camera_height` | 1280 × 720 | Capture resolution |
| `process_every_nth_captured_frame` | 6 | Skip N-1 frames between processing steps |
| `backend_origin` | `http://127.0.0.1:8001` | Backend URL |
| `auto_reset_on_init_detect_fail_streak` | 60 | Auto-reset after N consecutive detection failures |

### Backend — `backend/core/config.py`

| Field | Default | Description |
|---|---|---|
| `stockfish_path` | `""` | Full path to the Stockfish binary — **must be set** |
| `deep_depth` | 16 | Stockfish search depth |
| `deep_multipv` | 3 | Number of top lines to analyze |

---

## Camera Setup

The camera must be positioned directly above the board in a fixed, top-down orientation such that:

- **A1 is in the top-left corner** of the raw camera image
- **H8 is in the bottom-right corner** of the raw camera image

This orientation is assumed by the `raw_to_standard()` transform in `vision/pipeline/tracker.py`, which applies a 90° rotation and horizontal flip to convert from camera coordinates to standard chess coordinates (A1 = bottom-left).

The board detection algorithm (`findChessboard`) is robust to mild perspective tilt and rotation, but the A1/H8 corner assignment is fixed by the `raw_to_standard` transform — changing the camera orientation requires adjusting that transform.

---

## Robot Integration

The robot layer is **optional**. To integrate a physical robot arm:

1. Implement the abstract `RobotInterface` in `robot/interface.py`:
   - `execute_move(descriptor: dict)` — receives a structured move descriptor with all physical coordinates
   - `get_position() -> tuple[float, float, float]` — returns current end-effector position in mm

2. Run the calibration script once before each session:
   ```bash
   python -m robot.calibrate
   ```
   The script guides the operator to position the arm over **A1** and **H1**, records the coordinates, and saves them to `robot/calibration.json`. Optionally, the graveyard zone (where captured pieces are placed) can also be calibrated.

3. Use `robot.move_descriptor.build_move_descriptor(uci, board, calibration, graveyard)` to convert any UCI move into a physical move descriptor dict. The descriptor handles all move types: simple moves, captures (with graveyard slot allocation), castling (king + rook), and en passant.

The backend's `GET /api/robot/best-move` endpoint returns the Stockfish best move as a robot-compatible payload with row/col grid coordinates.

---

## Project Structure

```
_SAKKPROJEKT_FINAL/
│
├── chess_logic/               # Game state and move resolution
│   ├── types.py               # MoveGuess, MoveRecord, OccupancyResolveResult dataclasses
│   ├── game.py                # Board and Game classes (single source of truth)
│   ├── resolver.py            # board_to_occupancy, resolve_move_from_occupancy (exact + fuzzy)
│   └── stabilizer.py         # StateStabilizer — vision-to-chess handoff gating
│
├── vision/
│   ├── app/
│   │   ├── config.py          # AppConfig and LiveConfig dataclasses
│   │   ├── run_live.py        # Entry point: camera + live processor + preview window
│   │   └── debug_draw.py      # Overlay text and square classification dots
│   ├── board/
│   │   ├── find_chessboard.py # Saddle point detection + iterative homography fitting
│   │   └── squares.py         # Grid line refinement and square bounding box extraction
│   ├── models/
│   │   ├── weights/           # Model checkpoint (.pt file) — not in repo, add manually
│   │   └── occupancy_color_model.py  # ResNet18 wrapper (3-class: empty/white/black)
│   └── pipeline/
│       ├── tracker.py         # ChessVisionTracker — main processing pipeline
│       ├── batch_classifier.py # GPU batch inference for 64 squares
│       ├── board_detector.py  # detect_board_on_frame() — wraps find_chessboard
│       ├── square_diff.py     # Pixel-level diff for partial reclassification
│       └── profiler.py        # Optional timing instrumentation
│
├── backend/
│   ├── main.py                # FastAPI app factory + lifecycle events
│   ├── core/
│   │   ├── config.py          # Settings (Stockfish path, depth, CORS)
│   │   └── state.py           # BackendState — game + engine + WebSocket notifier
│   ├── api/
│   │   ├── routes.py          # REST endpoints + WebSocket /api/ws/state
│   │   └── schemas.py         # MoveRequest Pydantic model
│   └── services/
│       ├── engine_service.py  # EngineAnalysisService — Stockfish background thread
│       ├── robot_service.py   # UCI-to-robot-payload conversion
│       └── ws_service.py      # WebSocketHub — multi-client broadcaster
│
├── robot/
│   ├── interface.py           # Abstract RobotInterface (ABC)
│   ├── calibration.py         # Calibration — A1/H1 to all 64 squares (bilinear)
│   ├── graveyard.py           # Graveyard — sequential slot allocation for captured pieces
│   ├── move_descriptor.py     # build_move_descriptor() — UCI to physical coordinates
│   └── calibrate.py           # Interactive calibration CLI script
│
├── frontend/
│   ├── index.html             # Single-page app
│   ├── style.css              # Dark theme, eval bar, move list, toasts
│   ├── assets/chessboardjs-1.0.0/  # Chessboard.js library + ice piece images
│   └── js/
│       ├── main.js            # Entry point (calls bootstrapApp)
│       ├── app/
│       │   ├── app.js         # Main logic: WebSocket, rendering, user moves
│       │   ├── config.js      # Backend URL, timeouts, reconnect delays
│       │   ├── state.js       # Global currentState and connectionState store
│       │   └── selectors.js   # Signature functions for selective UI updates
│       ├── api/
│       │   ├── api.js         # REST: fetchState, postMove, postNewGame
│       │   └── socket.js      # StateSocket — WebSocket with auto-reconnect + ping
│       ├── ui/
│       │   ├── board.js       # Chessboard.js wrapper + move animation
│       │   ├── eval.js        # Evaluation bar rendering
│       │   ├── moves.js       # Move list (incremental or full render)
│       │   ├── toplines.js    # Engine top lines display
│       │   ├── status.js      # Connection + game status + side to move
│       │   └── toast.js       # Temporary notifications
│       └── utils/
│           ├── dom.js         # getEl, setText, emptyElement helpers
│           ├── format.js      # Score formatting, move number insertion
│           └── game.js        # isGameFinished, result/reason label helpers
│
├── collect_training_data.py   # Live data collection for training the classifier
├── split_train_val.py         # 80/20 train/val split for collected images
└── requirements.txt
```

---

## API Reference (brief)

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness check |
| GET | `/api/state` | Full game state (FEN, moves, top lines, PGN, …) |
| POST | `/api/new-game?fen=<FEN>` | Start a new game (optional starting FEN) |
| POST | `/api/move` | Apply a move: body `{"uci": "e2e4"}` |
| GET | `/api/robot/best-move` | Best move as robot-compatible payload |
| WS | `/api/ws/state` | Real-time state push; send `"ping"` to keep alive |
