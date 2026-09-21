"""
FastAPI application entry point.

Configures CORS, lifespan events (DB init, directory creation),
and mounts the API router.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings
from backend.models.database import init_db, close_db
from backend.api.routes import router

# ── Logging Setup ─────────────────────────────────────────────────────

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting Multimodal Multi-Agent News Fact-Checker backend")

    # Create data directories
    settings.ensure_directories()
    logger.info("Data directories ensured at: %s", settings.data_dir)

    # Initialize database
    await init_db(settings.sqlite_db_path)
    logger.info("Database initialized at: %s", settings.sqlite_db_path)

    yield

    # Shutdown
    await close_db()
    logger.info("Backend shutdown complete")


# ── FastAPI App ───────────────────────────────────────────────────────

app = FastAPI(
    title="Multimodal Multi-Agent News Fact-Checker",
    description=(
        "A research-grade multimodal multi-agent framework for real-time "
        "news verification. Performs claim-level fact checking through "
        "evidence-grounded cross-source reasoning."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Streamlit runs on localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router)

# Serve uploaded files statically (for image display in frontend)
from pathlib import Path

uploads_path = Path(settings.data_dir) / "uploads"
uploads_path.mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/uploads",
    StaticFiles(directory=str(uploads_path)),
    name="uploads",
)


# ── Direct Run ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
