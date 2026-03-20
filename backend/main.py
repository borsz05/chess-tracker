import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.core.config import settings
from backend.core.state import BackendState
from backend.services.ws_service import WebSocketHub


def create_app() -> FastAPI:
    app = FastAPI(title="Chess Backend API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ws_hub = WebSocketHub()
    app.state.ws_hub = ws_hub
    app.state.backend_state = None

    app.include_router(router)

    @app.on_event("startup")
    async def startup_event():
        app.state.ws_hub.set_loop(asyncio.get_running_loop())
        backend_state = BackendState(settings)
        backend_state.set_state_notifier(ws_hub.notify_state_changed)
        app.state.backend_state = backend_state

    @app.on_event("shutdown")
    def shutdown_event():
        backend_state = getattr(app.state, "backend_state", None)
        if backend_state is not None:
            backend_state.shutdown()

    return app


app = create_app()
