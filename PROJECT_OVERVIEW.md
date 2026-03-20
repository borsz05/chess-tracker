# Project Overview

## Main goal

The final system should do this:

1. Read a live camera image.
2. Detect and rectify the chessboard.
3. Classify all 64 squares as `empty`, `white`, or `black`.
4. Stabilize noisy frame-by-frame observations.
5. Infer the legal move from the board-state change.
6. Update the internal chess state.
7. Send the updated state to the frontend.
8. Later, expose the best Stockfish move for a robot arm.

## Core runtime flow

### Vision side

- `vision/app/run_live.py`
  Live camera loop and backend synchronization.
- `vision/pipeline/tracker.py`
  Main vision pipeline coordinator.
- `vision/pipeline/board_detector.py`
  Finds the board and computes the perspective transform.
- `vision/pipeline/batch_classifier.py`
  Classifies the board squares with the trained model.
- `vision/pipeline/square_diff.py`
  Helps avoid full reclassification on every processed frame.
- `vision/models/occupancy_color_model.py`
  Loads the trained PyTorch model.

### Chess logic

- `chess_logic/resolver.py`
  Converts occupancy changes into legal chess moves.
- `chess_logic/stabilizer.py`
  Filters noisy observations before move resolution.
- `chess_logic/game.py`
  Single source of truth for the internal chess state.

### Backend and UI

- `backend/main.py`
  FastAPI app setup.
- `backend/core/state.py`
  Holds game state and Stockfish analysis state.
- `backend/services/engine_service.py`
  Runs Stockfish analysis in the background.
- `backend/services/ws_service.py`
  Pushes state updates to the frontend.
- `frontend/`
  Browser UI that renders the board, moves, and engine lines.

## Important vs optional

### Important now

- `vision/`
- `chess_logic/`
- `backend/`
- `frontend/`
- `stockfish/stockfish-windows-x86-64-avx2.exe`
- `vision/models/weights/resnet18_best_szines_topdown_kepeken.pt`

### Optional or future-facing

- `backend/services/robot_service.py`
  Useful later for robot-arm integration, not needed for the current live demo.
- `backend/api/routes.py -> /api/robot/best-move`
  Future integration endpoint.
- `stockfish/src/`, `stockfish/wiki/`, `stockfish/*.md`, `stockfish/CITATION.cff`
  Not required at runtime if you only use the bundled executable.

## Simplifications already applied

- Removed old path hacks from the vision code.
- Switched to package-based imports.
- Removed unused debug and compatibility helpers.
- Removed one unused chess helper module.
- Delayed backend engine startup until FastAPI startup.
- Added `requirements.txt`.
- Added local `.venv` support and `.gitignore`.
