# System Architecture

This document describes the full data flow, component responsibilities, coordinate systems, container setup, and integration contracts for the Chess Vision + Robot Arm system.

---

## Component Map

```
┌─────────────────────────────────────────────────────────────────────┐
│  Host machine (Ubuntu 22.04)                                        │
│                                                                     │
│  ┌──────────┐   frame    ┌─────────────────────┐                   │
│  │  Webcam  │──────────▶│  vision/             │                   │
│  │ (V4L2)  │           │  LatestFrameCamera    │                   │
│  └──────────┘           │  LiveProcessor        │                   │
│                          │  ChessVisionTracker   │                   │
│                          └────────┬────────────┘                   │
│                     POST /api/move│                                 │
│                          ┌────────▼────────────┐                   │
│                          │  backend/   :8001    │                   │
│                          │  FastAPI + Stockfish │◀── WS ─── Browser│
│                          └────────┬────────────┘                   │
│                                   │ (developer must wire)          │
│                          ┌────────▼────────────┐                   │
│                          │  robot/              │                   │
│                          │  RobotImpl (HTTP)    │                   │
│                          └────────┬────────────┘                   │
│               POST /execute :8002 │                                 │
│  ┌────────────────────────────────▼───────────────────────────┐   │
│  │  Docker container: franka_ros2_humble                       │   │
│  │  ROS2 Humble  +  MoveIt2  +  chess_executor.py  :8002      │   │
│  └──────────────────────────────┬──────────────────────────────┘  │
└─────────────────────────────────┼───────────────────────────────────┘
                                  │ EtherCAT (libfranka)
                        ┌─────────▼──────────┐
                        │  Franka Research 3  │
                        └────────────────────┘
```

---

## Layer Descriptions

### vision/

**Entry point:** `python -m vision.app.run_live`

**Responsibilities:**
- Capture frames from the webcam in a background thread (`LatestFrameCamera`, buffer = 1)
- Detect board corners once at startup using the saddle-point algorithm (Hessian determinant maxima → contour fitting → iterative homography refinement)
- Warp every subsequent frame to a flat top-down view using the stored 3×3 homography matrix
- Classify all 64 squares per frame with ResNet18 batch inference (3 classes: empty / white-piece / black-piece)
- Apply pixel-diff-based partial reclassification — only re-infer squares that changed by more than a threshold
- Gate noisy outputs through `StateStabilizer` (majority vote over a sliding buffer, hysteresis)
- Resolve the stable occupancy grid to a legal UCI move via `resolve_move_from_occupancy`
- Push accepted moves to the backend via `BackendSyncClient → POST /api/move`

**Key classes:**

| Class | File | Role |
|---|---|---|
| `LatestFrameCamera` | `vision/app/run_live.py` | Thread-safe single-frame camera buffer |
| `LiveProcessor` | `vision/app/run_live.py` | Processing loop, backend sync, auto-reset |
| `ChessVisionTracker` | `vision/pipeline/tracker.py` | Main pipeline: detect → classify → stabilize → resolve |
| `StateStabilizer` | `chess_logic/stabilizer.py` | Vision→chess handoff gating |
| `OccupancyColorModel` | `vision/models/occupancy_color_model.py` | ResNet18 wrapper, CUDA/CPU |

**Coordinate systems:**

- Camera image: (row, col) pixels, origin top-left. A1 is at top-left, H8 at bottom-right.
- Warped image: flat 1632×1632 px (17 × 96 px per square including padding). Same A1=top-left orientation.
- Standard chess grid (internal): A1=bottom-left. `raw_to_standard()` applies rot90+fliplr.
- The homography matrix `M` (3×3 float64) maps raw frame pixels → warped board pixels (via `cv2.WARP_INVERSE_MAP`).

---

### chess_logic/

**Responsibilities:**
- Authoritative game state (`Board`, `Game`): move history, FEN, legal move generation
- `board_to_occupancy()`: converts python-chess `Board` → 8×8 int array (0=empty, 1=white, 2=black)
- `resolve_move_from_occupancy()`: finds the unique legal move consistent with observed vs. expected occupancy; supports exact matching and fuzzy matching (up to N noisy cells, confidence-weighted cost)
- `StateStabilizer`: vote-and-hysteresis gate between raw classifications and accepted occupancy

---

### backend/

**Entry point:** `uvicorn backend.main:app --port 8001`

**Responsibilities:**
- Single authoritative `Game` instance for the session
- REST endpoints: new-game, apply-move, get-state, robot best-move
- `EngineAnalysisService`: Stockfish subprocess in a background thread; analysis requests queued, results stored in `BackendState`
- `WebSocketHub`: broadcast state diffs to all connected browser clients

**Port:** 8001 (host, native Python process)

**State notifier:** After any state change, `BackendState` calls `ws_hub.notify_state_changed()` which schedules an async broadcast to all WebSocket clients.

**Robot endpoint:** `GET /api/robot/best-move` returns the Stockfish best move as a grid-coordinate payload (`row`, `col`). This is a convenience endpoint for the frontend — it does **not** trigger physical execution. Full physical execution requires calling `build_move_descriptor` + `RobotImpl.execute_move` (see robot/ below).

---

### robot/

**Responsibilities:**
- `Calibration`: maps any chess square name → physical (x, y) in mm using A1 and H8 reference points
- `Graveyard`: sequential slot allocator for captured pieces; position computed from start + step offset
- `build_move_descriptor()`: converts a UCI string + python-chess Board state → typed action descriptor with all physical coordinates for every sub-move in the sequence
- `RobotInterface` (ABC): contract for any robot arm driver
- `RobotImpl` (`robot/impl/__init__.py`): concrete implementation — HTTP client to `chess_executor.py` on port 8002
- `robot/calibrate.py`: interactive CLI for teaching A1 and H8 corners + optional graveyard calibration

**Coordinate system:** All positions in millimetres, in the robot's base frame. Only (x, y) are stored; Z heights are constants in `executor.py` (`Z_TRAVEL=0.25m`, `Z_PICK=0.04m`, `Z_PLACE=0.03m`).

**Calibration math (A1 + H8 → all 64 squares):**
```
(dx, dy) = H8 − A1
file_step  = ((dx+dy)/14,  (dy−dx)/14)
rank_step  = (−file_step_y, file_step_x)   # 90° CCW
square(col, row) = A1 + col·file_step + row·rank_step
col ∈ [0..7] = a..h,  row ∈ [0..7] = rank1..rank8
```

**Move descriptor types and physical sequences:**

| Type | Sequence |
|---|---|
| `simple` | pick(from) → place(to) |
| `capture` | pick(captured) → place(graveyard); pick(from) → place(to) |
| `en_passant` | pick(captured pawn, different square) → place(graveyard); pick(from) → place(to) |
| `castling` | pick(king_from) → place(king_to); pick(rook_from) → place(rook_to) |

**Missing integration:** There is currently no code that automatically triggers `RobotImpl.execute_move()` when the vision pipeline detects a move. To wire this in, add a callback or hook in `LiveProcessor._handle_successful_process()` that builds and dispatches the descriptor.

---

### franka/ (Docker container)

**Image:** `ros:humble-ros-base` (Ubuntu 22.04)

**What runs inside:**
1. `ros2 launch franka_fr3_moveit_config moveit.launch.py` — MoveIt2 + Franka driver (or fake hardware for simulation)
2. `python3 /ros2_ws/chess_executor.py` — HTTP server on port 8002

**chess_executor.py architecture:**
- Initializes `rclpy`, creates a `Node("chess_executor")`
- Creates `MoveIt2` instance targeting group `fr3_arm`, joints `fr3_joint1..7`, EEF `fr3_hand_tcp`
- Creates `MoveIt2Gripper` for `fr3_finger_joint1/2`
- Creates `TransformListener` on `Buffer` for TF lookups (position feedback)
- Spins ROS2 executor in a daemon thread
- Serves `HTTPServer` on `0.0.0.0:8002` (blocking)
- All POST `/execute` requests are serialized with a threading lock

**Robot communication:**
- Protocol: MoveIt2 action interface (ROS2 actions over DDS/RMW)
- Trajectory planning: OMPL (default planner)
- Cartesian moves: `MoveIt2.move_to_pose(position=[x,y,z], quat_xyzw=[1,0,0,0])` — gripper always pointing straight down
- Position feedback: TF lookup `fr3_link0 → fr3_hand_tcp` via `tf2_ros`

**Docker configuration (`docker-compose.yml`):**
- `network_mode: "host"` — no port mapping; executor port 8002 is directly accessible on the host
- `privileged: true` — required for EtherCAT access
- `cap_add: SYS_NICE` — required for realtime thread priority
- `ulimits: rtprio=99, rttime=-1` — realtime scheduling
- Volume `/dev:/dev` — Ethernet NIC passthrough for EtherCAT
- Volume `/tmp/.X11-unix` + `DISPLAY` env — X11 for RViz2 (set `xhost +local:docker` on host)
- Volume `${HOME}/franka_ros2_ws/src:/ros2_ws/src` — external ROS2 workspace sources (must exist on host)
- Volume `./limits.conf:/etc/security/limits.conf` — realtime limits

**Ports used:**
| Port | Service | Where |
|---|---|---|
| 8001 | FastAPI backend | Host (native Python) |
| 8002 | chess_executor HTTP | Docker container (host network) |

---

## Startup Sequence

The correct startup order is enforced by convention and partially by code:

```
1. docker-compose build       (once, ~20 min)
2. uvicorn backend.main:app   (FastAPI, port 8001)
3. ./run.sh [sim|real <IP>]   (Docker + MoveIt2 + executor, port 8002)
4. python -m robot.calibrate  (teach A1 + H8; saves calibration.json)
5. python -m vision.app.run_live   (checks calibration.json; asks confirmation)
6. open http://localhost:8000  (frontend)
```

Steps 4 → 5 ordering is enforced at step 5 startup: `run_live.py` checks that `robot/calibration.json` exists and prompts the operator to confirm the board is clear. Pass `--no-robot` to skip this check for vision-only operation.

---

## Thread Model

| Thread | Owner | Purpose |
|---|---|---|
| Camera reader | `LatestFrameCamera._reader_loop` | Continuous frame capture; writes to single-element buffer under lock |
| Vision processor | `LiveProcessor._worker_loop` | Reads every Nth frame; runs tracker; pushes to backend |
| Main thread | `run_live.py main()` | Preview window render loop; keyboard input |
| FastAPI/uvicorn | asyncio event loop | REST + WebSocket server |
| Stockfish engine | `EngineAnalysisService._worker` | Background analysis; queue-based |
| ROS2 executor | `chess_executor.py` daemon thread | Spins MoveIt2 callbacks |
| HTTP server | `chess_executor.py main thread` | Blocking `HTTPServer.serve_forever()` |

All shared state is protected by `threading.Lock`. The vision tracker (`ChessVisionTracker`) is accessed under `_tracker_lock` in `LiveProcessor`.

---

## Key Configuration Files

| File | What to set |
|---|---|
| `backend/core/config.py` | `stockfish_path` — full path to Stockfish binary |
| `vision/app/config.py` | `AppConfig.weights_path` — path to ResNet18 checkpoint |
| `vision/app/run_live.py` | `LiveConfig.camera_index`, `backend_origin` |
| `robot/impl/__init__.py` | `_EXECUTOR_URL` — default `http://localhost:8002` |
| `robot/calibrate.py` | `ROBOT_IMPL` import — point at your concrete driver class |
