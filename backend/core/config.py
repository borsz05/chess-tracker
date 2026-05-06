from dataclasses import dataclass, field


@dataclass(slots=True)
class Settings:
    start_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    backend_port: int = 8001

    stockfish_path: str = "/usr/games/stockfish"
    deep_depth: int = 16
    deep_multipv: int = 3

    # Robot integration.
    # robot_enabled: set False to disable the robot even if the executor is running.
    # robot_color: which side the robot plays ("black" or "white").
    robot_enabled: bool = True
    robot_color: str = "black"

    cors_origins: list[str] = field(default_factory=lambda: [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ])


settings = Settings()