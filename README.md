# Chess Vision + Robot Arm

A camera-based chess game tracker with a Franka Research 3 robotic arm that physically executes moves on the board. A webcam positioned above the board detects the board state in real time using a ResNet18 classifier; when a move is detected it is relayed to the backend, analyzed by Stockfish, and broadcast to a live browser interface. The robot arm is a fully integrated core component — not optional infrastructure — and executes every physical move on the board.

---

## System Architecture

```
Webcam (top-down, fixed)
        │
        ▼
  [vision/]          Board detection (saddle points + homography, once at startup)
                     Square occupancy classification (ResNet18, per frame)
                     Move stabilization + legal-move resolution
        │  HTTP POST /api/move
        ▼
  [backend/]         FastAPI REST + WebSocket server  (port 8001)
                     Stockfish analysis (background thread)
                     State broadcast to all browser clients
        │
        ▼
  [frontend/]        Browser UI — live board, eval bar, engine lines, move list

  [robot/]           Move descriptor builder (UCI → physical coordinates)
                     HTTP client → executor
        │  HTTP POST /execute → localhost:8002
        ▼
  [franka/ Docker]   ROS2 Humble + MoveIt2 + chess_executor.py  (port 8002)
        │  EtherCAT
        ▼
  Franka Research 3
```

Data flow summary:
1. `LatestFrameCamera` captures frames in a background thread (buffer = 1, always latest).
2. `LiveProcessor` processes every 6th captured frame in a second thread.
3. `ChessVisionTracker` detects board corners once at startup (saddle points + homography), then warps each frame and classifies all 64 squares with ResNet18 batch inference.
4. `StateStabilizer` gates noisy classifications; `resolve_move_from_occupancy` identifies the legal move matching the observed occupancy.
5. The accepted move is pushed to the backend (`POST /api/move`).
6. The backend stores the move, schedules Stockfish analysis, and pushes the updated state to all WebSocket clients.
7. The frontend re-renders only the changed UI components.
8. **The robot layer is responsible for physical execution** — see [Robot Execution](#robot-execution) below.

---

## Hardware Requirements

| Component | Requirement |
|---|---|
| Webcam | Fixed top-down mount directly above the board. **A1 must be in the top-left corner** of the raw camera image; **H8 in the bottom-right**. |
| Robot arm | Franka Research 3 (FR3). Communicates over Ethernet. |
| Host machine | Ubuntu 22.04 LTS. Docker + docker-compose installed. Realtime kernel recommended (`linux-lowlatency`) for the Franka EtherCAT driver. |

---

## Dependencies

### Python (host)

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `fastapi`, `uvicorn[standard]`, `pydantic` | Backend server |
| `python-chess` | Move generation and validation |
| `numpy`, `opencv-python` | Image processing, warp, diffs |
| `torch`, `torchvision` | ResNet18 square classifier |
| `requests` | Robot HTTP client (`robot/impl/`) |

### System packages

```bash
sudo apt install stockfish tmux docker.io docker-compose
```

### Model weights

The ResNet18 checkpoint is not included in this repository. Obtain it and place it at:

```
vision/models/weights/resnet18_best_topdown.pt
```

### Training data and model training (`tools/`)

| Script | Purpose |
|---|---|
| `tools/collect_fen_dataset.py` | Collect training ROIs from the live camera, labelled automatically from a known FEN (occupancy **and** piece type). Uses the exact live crop path (`detect_board_on_frame` → warp → `crop_with_context(context=0.50)`). Output is compatible with the `train_new/val_new/{black,empty,white}` layout of the [existing dataset](https://github.com/borsz05/sakk_modelltanitas). |
| `tools/train_square_classifier.py` | Colab-ready trainer for the hybrid multi-task square classifier (3-class colour head + 7-class piece-type head, architecture in `vision/models/square_net.py`). `--benchmark-only` measures the candidate backbones on CPU under ONNX Runtime; the header documents the measured decision. |
| `tools/fen_labels.py` | FEN → raw (camera) grid labels, via the inverse of the pipeline's `raw_to_standard`. |
| `tools/dump_live_rois.py` | Dump live ROIs with the current model's predictions, for visual comparison with the training set. |

```bash
# fixed exposure / white balance is essential — auto-exposure makes black pieces flicker in brightness
python -m tools.collect_fen_dataset --session morning_window --fen-file positions.txt --exposure 250 --wb-temp 4600
python tools/train_square_classifier.py --benchmark-only
python tools/train_square_classifier.py --data-root ../sakk_modelltanitas --data-root data_fen --eval-dir data_fen --arch mobilenet_v3_small
python -m pytest tests
```

### Robot (Docker container)

Build once — this pulls ROS2 Humble, libfranka, franka\_description, pymoveit2, and MoveIt2:

```bash
cd franka && docker-compose build
```

The build requires internet access and takes ~15–30 minutes.

---

## Startup Instructions

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Stockfish path

Edit `backend/core/config.py` and set `stockfish_path` to the actual binary location:

```python
stockfish_path: str = "/usr/games/stockfish"   # adjust if installed elsewhere
```

Verify: `which stockfish` or `ls /usr/games/stockfish`.

### 3. Start the backend server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8001
```

The backend initializes the game state and immediately queues a Stockfish analysis of the starting position.

### 4. Start the robot container

```bash
# Simulation (no physical robot — MoveIt2 uses fake hardware):
./run.sh sim

# Real robot (provide the robot's Ethernet IP):
./run.sh real 192.168.1.1
```

`run.sh` starts a tmux session with two panes:
- Top pane: MoveIt2 launch inside the Docker container
- Bottom pane: `chess_executor.py` HTTP server on port **8002** (starts 8 s after MoveIt2)

Wait until the bottom pane prints `[chess_executor] Robot kész. HTTP szerver: port 8002` before proceeding.

### 5. Run calibration

**Before starting the vision pipeline**, physically position the robot arm over each calibration corner and record its coordinates. The board and the robot arm must be in their final positions before this step.

```bash
python -m robot.calibrate
```

The script guides you through:
1. Move the gripper to the center of **A1** (white queen-side rook) → press Enter
2. Move the gripper to the center of **H8** (black king-side rook) → press Enter
3. Optional: teach the graveyard zone (first and second captured-piece slots)

Calibration is saved to `robot/calibration.json` and persists across restarts. Re-run whenever the board or robot base position changes.

### 6. Start the vision pipeline

**Clear the board of hands and the robot arm first.** Then:

```bash
python -m vision.app.run_live
```

The startup prompt will confirm that calibration is complete and the board is clear before the camera begins processing. Pass `--no-robot` to skip the calibration check when running vision-only (no robot connected).

The pipeline will:
1. Detect board corners (one-time saddle-point + homography fit)
2. Collect 3 baseline frames and verify the board matches the expected starting position (Hamming distance ≤ 2 squares)
3. Once initialized, continuously classify squares, detect moves, and push them to the backend

**Keyboard shortcuts in the preview window:**
- `q` — quit
- `r` — reset tracker and sync a new game to the backend

### 7. Open the frontend

```bash
python -m http.server 8000 --directory frontend
```

Open `http://localhost:8000` in any modern browser. The page connects to the backend via WebSocket and displays the live board, Stockfish evaluation bar, engine lines, and move list.

---

## Calibration Details

Two diagonal corners establish the robot's coordinate frame for the board:

| Point | Square | Description |
|---|---|---|
| A1 | a-file, rank 1 | White queen-side rook starting square |
| H8 | h-file, rank 8 | Black king-side rook starting square |

From these two points all 64 square positions are derived by solving for file and rank unit vectors:

```
Given (dx, dy) = H8 − A1:
  file_step = ((dx+dy)/14, (dy−dx)/14)
  rank_step = (−file_step_y, file_step_x)   # 90° CCW rotation
  square(col, row) = A1 + col·file_step + row·rank_step
```

The **graveyard** (where captured pieces are deposited beside the board) is calibrated by teaching the first two graveyard slots; all subsequent slots are interpolated in the same direction. If graveyard calibration is skipped, a default offset of 30 mm per slot along the X axis is used.

Calibration data is stored in `robot/calibration.json`:
```json
{
  "a1": [x_mm, y_mm],
  "h8": [x_mm, y_mm],
  "graveyard_start": [x_mm, y_mm],
  "graveyard_step": [dx_mm, dy_mm]
}
```

---

## Robot Execution

### Protocol

`robot/executor.py` runs inside the Docker container (ROS2 Humble + MoveIt2) and exposes an HTTP API on port **8002**:

| Method | Path | Body | Description |
|---|---|---|---|
| GET | `/health` | — | Liveness check |
| GET | `/position` | — | End-effector position in mm (reads TF `fr3_link0 → fr3_hand_tcp`) |
| POST | `/execute` | move descriptor (see below) | Execute a full chess move |
| POST | `/gripper/open` | — | Open gripper |
| POST | `/gripper/close` | — | Close gripper |
| POST | `/move` | `{"x": mm, "y": mm, "z": mm}` | Direct Cartesian move |

### Move descriptor

`robot.move_descriptor.build_move_descriptor(uci, board, calibration, graveyard)` converts any UCI move string into a physical action descriptor. Example usage:

```python
import chess
from robot.calibration import Calibration
from robot.graveyard import Graveyard
from robot.move_descriptor import build_move_descriptor
from robot.impl import RobotImpl

cal = Calibration.load()
graveyard = Graveyard.from_calibration_file()
board = chess.Board()
robot = RobotImpl()

descriptor = build_move_descriptor("e2e4", board, cal, graveyard)
robot.execute_move(descriptor)
```

Supported move types:

| Type | Physical sequence |
|---|---|
| `simple` | pick piece from source → place at destination |
| `capture` | pick captured piece → place in graveyard; then pick moving piece → place at destination |
| `en_passant` | pick captured pawn (different square) → graveyard; move pawn |
| `castling` | move king; then move rook |

### Physical parameters (`executor.py`)

| Constant | Value | Purpose |
|---|---|---|
| `Z_TRAVEL` | 0.25 m | Safe clearance height during travel |
| `Z_PICK` | 0.04 m | Height to grip a piece |
| `Z_PLACE` | 0.03 m | Height to release a piece |
| `VELOCITY` | 0.3 m/s | MoveIt2 max velocity |
| `ACCELERATION` | 0.3 m/s² | MoveIt2 max acceleration |

### Extending to a different robot arm

Implement `robot.interface.RobotInterface` for your robot:

```python
from robot.interface import RobotInterface

class MyRobot(RobotInterface):
    def execute_move(self, descriptor: dict) -> None:
        ...   # interpret descriptor["type"] + physical coordinates

    def get_position(self) -> tuple[float, float, float]:
        ...   # return end-effector (x, y, z) in mm
```

Edit the `ROBOT_IMPL` import in `robot/calibrate.py` to point at your class.

---

## Container Setup

The Franka integration runs in a Docker container (`franka/`):

| File | Purpose |
|---|---|
| `franka/Dockerfile` | ROS2 Humble base; installs libfranka, franka\_description, pymoveit2, MoveIt2 stack via `rosdep` |
| `franka/docker-compose.yml` | Host networking, privileged, X11 forwarding, `/dev` passthrough, realtime priority (rtprio=99) |
| `franka/franka_entrypoint.sh` | Container entrypoint — sources ROS2 setup, colcon builds the workspace |
| `franka/dependency.repos` | VCS repo list: franka\_description 1.6.1, libfranka 0.20.4, pymoveit2, olvx\_descriptions\_module |
| `franka/limits.conf` | Realtime priority limits |

**Volumes mounted at runtime:**
- `${HOME}/franka_ros2_ws/src` → `/ros2_ws/src` — ROS2 workspace (must exist on host before first run)
- `/tmp/.X11-unix` → X11 display for MoveIt2/RViz2 visualization
- `/dev` → hardware device access (EtherCAT NIC for Franka)

The container uses `network_mode: "host"` — all ports are shared with the host. Port 8002 is used by `chess_executor.py`.

---

## Configuration Reference

### `vision/app/config.py` — `AppConfig`

| Field | Default | Description |
|---|---|---|
| `weights_path` | `vision/models/weights/resnet18_best_topdown.pt` | ResNet18 checkpoint |
| `cell` | 96 | Pixel size of each square in warped image |
| `context` | 0.50 | Padding ratio around each square crop fed to the model |
| `init_buffer_frames` | 3 | Frames averaged for initial position baseline |
| `init_max_dist` | 2 | Max Hamming distance from expected start during init |
| `partial_reclassify` | True | Only re-classify squares that changed (faster) |
| `full_reclassify_interval` | 30 | Force full classification every N frames |

### `vision/app/run_live.py` — `LiveConfig`

| Field | Default | Description |
|---|---|---|
| `camera_index` | 0 | OpenCV camera index |
| `camera_width` / `camera_height` | 1280 × 720 | Capture resolution |
| `process_every_nth_captured_frame` | 6 | Skip N−1 frames between processing steps |
| `backend_origin` | `http://127.0.0.1:8001` | Backend URL |
| `auto_reset_on_init_detect_fail_streak` | 60 | Auto-reset after N consecutive detection failures |

### `backend/core/config.py` — `Settings`

| Field | Default | Description |
|---|---|---|
| `stockfish_path` | `/usr/games/stockfish` | Full path to Stockfish binary |
| `deep_depth` | 16 | Stockfish search depth |
| `deep_multipv` | 3 | Number of top lines to return |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness check |
| GET | `/api/state` | Full game state (FEN, moves, top lines, PGN) |
| POST | `/api/new-game?fen=<FEN>` | Start new game (optional FEN) |
| POST | `/api/move` | Apply move: `{"uci": "e2e4"}` |
| GET | `/api/robot/best-move` | Best move as robot grid-coordinate payload |
| WS | `/api/ws/state` | Real-time state push; send `"ping"` to keep alive |

---

## Known Limitations

1. **Board detection is one-time.** The homography is computed once at startup and is fixed for the session. If the camera or board shifts during play, press `r` in the preview window to reset.

2. **Robot execution is not automatically triggered.** The vision pipeline detects moves and pushes them to the backend via HTTP. The integration layer that calls `build_move_descriptor` → `RobotImpl.execute_move` on each detected move must be wired in by the developer. The building blocks are all present in the `robot/` module.

3. **`${HOME}/franka_ros2_ws/src` must exist** before running `docker-compose`. Create it: `mkdir -p ~/franka_ros2_ws/src`.

4. **Model weights not included.** Obtain `resnet18_best_topdown.pt` and place it in `vision/models/weights/` before starting the vision pipeline.

5. **Calibration file format changed.** If you have an existing `robot/calibration.json` from before this update (which stored key `"h1"`), delete it and re-run `python -m robot.calibrate`. The new format stores key `"h8"`.
