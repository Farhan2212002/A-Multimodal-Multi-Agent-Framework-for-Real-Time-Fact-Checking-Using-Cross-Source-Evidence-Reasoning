"""
FastAPI API routes for the Multimodal Multi-Agent News Fact-Checker.

Endpoints:
  POST /api/fact-check/text     — Text input fact checking
  POST /api/fact-check/image    — Image input fact checking
  POST /api/fact-check/mixed    — Image + text fact checking
  POST /api/fact-check/url      — URL fact checking
  GET  /api/fact-check/{id}     — Retrieve result
  GET  /api/fact-check/{id}/report — Download report (json/txt/pdf)
  GET  /api/fact-checks         — List recent fact checks
  DELETE /api/fact-check/{id}   — Delete a fact check
  GET  /api/health              — Health check
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from backend.config import get_settings
from backend.models.schemas import (
    FactCheckReport,
    FactCheckResultResponse,
    FactCheckState,
    FactCheckStatusResponse,
    TextFactCheckRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory store for running/completed fact checks
# In production, this would be Redis or similar
_fact_check_states: dict[str, FactCheckState] = {}


# ── Helper: Run fact check in background ─────────────────────────────


async def _run_fact_check(
    fact_check_id: str,
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    url: Optional[str] = None,
) -> None:
    """Execute fact-check pipeline in a background task."""
    from backend.orchestrator import FactCheckOrchestrator

    settings = get_settings()

    async def progress_callback(
        agent_name: str, progress: float, status: str
    ) -> None:
        if fact_check_id in _fact_check_states:
            _fact_check_states[fact_check_id].current_agent = agent_name
            _fact_check_states[fact_check_id].progress_percent = progress
            _fact_check_states[fact_check_id].status = (
                "running" if status != "error" else "error"
            )

    orchestrator = FactCheckOrchestrator(
        settings=settings, progress_callback=progress_callback
    )

    # Initialize placeholder state
    _fact_check_states[fact_check_id] = FactCheckState(
        fact_check_id=fact_check_id,
        original_text=text,
        image_path=image_path,
        url=url,
        status="running",
    )

    try:
        result = await orchestrator.run(
            text=text, image_path=image_path, url=url
        )
        # Update fact_check_id to match what orchestrator assigned
        result.fact_check_id = fact_check_id
        _fact_check_states[fact_check_id] = result
    except Exception as e:
        logger.error("Background fact-check failed: %s", e, exc_info=True)
        if fact_check_id in _fact_check_states:
            _fact_check_states[fact_check_id].status = "error"
            _fact_check_states[fact_check_id].error_message = str(e)


# ── Health Check ─────────────────────────────────────────────────────


@router.get("/api/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model": settings.openai_model,
        "version": "1.0.0",
    }


# ── Text Fact Check ──────────────────────────────────────────────────


@router.post("/api/fact-check/text", tags=["Fact Check"])
async def fact_check_text(
    request: TextFactCheckRequest, background_tasks: BackgroundTasks
):
    """
    Submit text for fact checking.

    The fact check runs asynchronously. Poll /api/fact-check/{id} for status.
    """
    fact_check_id = f"fc-{uuid.uuid4().hex[:12]}"
    logger.info("Received text fact-check request: %s", fact_check_id)

    background_tasks.add_task(
        _run_fact_check, fact_check_id=fact_check_id, text=request.text
    )

    return FactCheckStatusResponse(
        fact_check_id=fact_check_id,
        status="running",
        progress_percent=0.0,
        current_agent="Queued",
    )


# ── Image Fact Check ─────────────────────────────────────────────────


@router.post("/api/fact-check/image", tags=["Fact Check"])
async def fact_check_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    language: str = Form("en"),
):
    """
    Submit an image for fact checking.

    Supports JPEG, PNG, GIF, WebP, BMP.
    """
    # Validate file type
    allowed_types = {
        "image/jpeg", "image/png", "image/gif",
        "image/webp", "image/bmp",
    }
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {image.content_type}. "
                   f"Supported: {', '.join(allowed_types)}",
        )

    fact_check_id = f"fc-{uuid.uuid4().hex[:12]}"
    settings = get_settings()

    # Save uploaded image
    image_content = await image.read()
    safe_filename = f"{fact_check_id}_{image.filename}"
    image_path = settings.uploads_dir / safe_filename
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_content)

    logger.info(
        "Received image fact-check: %s (%s, %d bytes)",
        fact_check_id, image.filename, len(image_content),
    )

    background_tasks.add_task(
        _run_fact_check,
        fact_check_id=fact_check_id,
        image_path=str(image_path),
    )

    return FactCheckStatusResponse(
        fact_check_id=fact_check_id,
        status="running",
        progress_percent=0.0,
        current_agent="Queued",
    )


# ── Mixed Fact Check (Image + Text) ──────────────────────────────────


@router.post("/api/fact-check/mixed", tags=["Fact Check"])
async def fact_check_mixed(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    image: UploadFile = File(...),
    language: str = Form("en"),
):
    """Submit both text and image for joint fact checking."""
    fact_check_id = f"fc-{uuid.uuid4().hex[:12]}"
    settings = get_settings()

    # Save uploaded image
    image_content = await image.read()
    safe_filename = f"{fact_check_id}_{image.filename}"
    image_path = settings.uploads_dir / safe_filename
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_content)

    logger.info("Received mixed fact-check: %s", fact_check_id)

    background_tasks.add_task(
        _run_fact_check,
        fact_check_id=fact_check_id,
        text=text,
        image_path=str(image_path),
    )

    return FactCheckStatusResponse(
        fact_check_id=fact_check_id,
        status="running",
        progress_percent=0.0,
        current_agent="Queued",
    )





# ── Get Fact Check Status/Result ─────────────────────────────────────


@router.get("/api/fact-check/{fact_check_id}", tags=["Fact Check"])
@router.get("/api/fact-check/{fact_check_id}/status", tags=["Fact Check"])
async def get_fact_check(fact_check_id: str):
    """
    Get the status or result of a fact check.

    Returns full results when the fact check is completed.
    """
    state = _fact_check_states.get(fact_check_id)

    if state is None:
        # Try loading from database
        try:
            from backend.models.database import get_session_direct, FactCheckRecord
            from sqlalchemy import select

            session = await get_session_direct()
            try:
                result = await session.execute(
                    select(FactCheckRecord).where(
                        FactCheckRecord.id == fact_check_id
                    )
                )
                record = result.scalar_one_or_none()
                if record and record.report_json:
                    report = FactCheckReport.model_validate_json(
                        record.report_json
                    )
                    return FactCheckResultResponse(
                        fact_check_id=fact_check_id,
                        status=record.status or "completed",
                        report=report,
                    )
                elif record:
                    return FactCheckStatusResponse(
                        fact_check_id=fact_check_id,
                        status=record.status or "unknown",
                        error_message=record.error_message,
                    )
            finally:
                await session.close()
        except Exception:
            pass

        raise HTTPException(status_code=404, detail="Fact check not found")

    if state.status in ("running", "pending"):
        return FactCheckStatusResponse(
            fact_check_id=fact_check_id,
            status=state.status,
            progress_percent=state.progress_percent,
            current_agent=state.current_agent,
        )

    return FactCheckResultResponse(
        fact_check_id=fact_check_id,
        status=state.status,
        report=state.report,
        error_message=state.error_message,
    )


# ── Get Report ───────────────────────────────────────────────────────


@router.get("/api/fact-check/{fact_check_id}/report", tags=["Report"])
async def get_report(fact_check_id: str, format: str = "json"):
    """
    Download the fact-check report.

    Args:
        format: Output format — json, txt, or pdf.
    """
    state = _fact_check_states.get(fact_check_id)
    if state is None or state.report is None:
        raise HTTPException(
            status_code=404,
            detail="Report not found or fact check not completed",
        )

    report = state.report
    settings = get_settings()

    if format == "json":
        return JSONResponse(
            content=json.loads(report.model_dump_json()),
            headers={
                "Content-Disposition": f'attachment; filename="report_{fact_check_id}.json"'
            },
        )

    elif format == "txt":
        from backend.services.reporting import generate_text_report

        text_report = generate_text_report(report, language=report.language)
        report_path = settings.reports_dir / f"{fact_check_id}.txt"
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text_report, encoding="utf-8")
        return FileResponse(
            path=str(report_path),
            filename=f"report_{fact_check_id}.txt",
            media_type="text/plain",
        )

    elif format == "pdf":
        from backend.services.reporting import generate_pdf_report

        report_path = str(
            settings.reports_dir / f"{fact_check_id}.pdf"
        )
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        generate_pdf_report(report, report_path, language=report.language)
        return FileResponse(
            path=report_path,
            filename=f"report_{fact_check_id}.pdf",
            media_type="application/pdf",
        )

    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported format: {format}"
        )


# ── List Recent Fact Checks ──────────────────────────────────────────


@router.get("/api/fact-checks", tags=["Fact Check"])
async def list_fact_checks(limit: int = 20):
    """List recent fact checks from the database."""
    try:
        from backend.models.database import get_session_direct, FactCheckRecord
        from sqlalchemy import select

        session = await get_session_direct()
        try:
            result = await session.execute(
                select(FactCheckRecord)
                .order_by(FactCheckRecord.created_at.desc())
                .limit(limit)
            )
            records = result.scalars().all()
            return [
                {
                    "fact_check_id": r.id,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "input_type": r.input_type,
                    "original_text": (
                        r.original_text[:200] + "..."
                        if r.original_text and len(r.original_text) > 200
                        else r.original_text
                    ),
                    "overall_verdict": r.overall_verdict,
                    "overall_confidence": r.overall_confidence,
                    "status": r.status,
                }
                for r in records
            ]
        finally:
            await session.close()
    except Exception as e:
        logger.error("Failed to list fact checks: %s", e)
        return []


# ── Delete Fact Check ────────────────────────────────────────────────


@router.delete("/api/fact-check/{fact_check_id}", tags=["Fact Check"])
async def delete_fact_check(fact_check_id: str):
    """Delete a fact check and all associated data (privacy)."""
    # Remove from in-memory store
    _fact_check_states.pop(fact_check_id, None)

    # Remove from database
    try:
        from backend.models.database import get_session_direct, FactCheckRecord
        from sqlalchemy import select, delete

        session = await get_session_direct()
        try:
            await session.execute(
                delete(FactCheckRecord).where(
                    FactCheckRecord.id == fact_check_id
                )
            )
            await session.commit()
        finally:
            await session.close()
    except Exception as e:
        logger.error("DB deletion failed: %s", e)

    # Remove associated files
    try:
        from backend.services.storage import delete_fact_check_files

        await delete_fact_check_files(fact_check_id)
    except Exception as e:
        logger.error("File deletion failed: %s", e)

    return {"status": "deleted", "fact_check_id": fact_check_id}
