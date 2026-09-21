"""
Fact-Check Orchestrator — central controller for the multi-agent pipeline.

Manages FactCheckState lifecycle, routes inputs to the appropriate pipeline
(text-only, image, mixed), and coordinates agent execution in sequence.
Supports ablation experiments via feature flags in Settings.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Coroutine, Optional

from backend.config import Settings, get_settings
from backend.models.schemas import (
    AgentTraceEntry,
    ClaimType,
    EvidenceGraph,
    FactCheckReport,
    FactCheckState,
    InputAnalysis,
    InputType,
    Verdict,
)

logger = logging.getLogger(__name__)


# Type alias for the progress callback
ProgressCallback = Optional[Callable[[str, float, str], Coroutine[Any, Any, None]]]


class FactCheckOrchestrator:
    """
    Orchestrates the multi-agent fact-checking pipeline.

    Decides which agents to run based on input type and ablation flags,
    passes structured state between agents, and manages execution tracing.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        progress_callback: ProgressCallback = None,
    ):
        self.settings = settings or get_settings()
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(self.__class__.__name__)

        # Lazy agent initialization
        self._agents_initialized = False
        self._input_agent = None
        self._claim_agent = None
        self._image_agent = None
        self._research_agent = None
        self._evidence_agent = None
        self._contradiction_agent = None
        self._temporal_agent = None
        self._judge_agent = None
        self._report_agent = None

    def _init_agents(self) -> None:
        """Initialize all agent instances (lazy, called once)."""
        if self._agents_initialized:
            return

        from backend.agents.input_agent import InputAgent
        from backend.agents.claim_agent import ClaimAgent
        from backend.agents.image_agent import ImageAgent
        from backend.agents.research_agent import ResearchAgent
        from backend.agents.evidence_agent import EvidenceAgent
        from backend.agents.contradiction_agent import ContradictionAgent
        from backend.agents.temporal_agent import TemporalAgent
        from backend.agents.judge_agent import JudgeAgent
        from backend.agents.report_agent import ReportAgent

        self._input_agent = InputAgent(self.settings)
        self._claim_agent = ClaimAgent(self.settings)
        self._image_agent = ImageAgent(self.settings)
        self._research_agent = ResearchAgent(self.settings)
        self._evidence_agent = EvidenceAgent(self.settings)
        self._contradiction_agent = ContradictionAgent(self.settings)
        self._temporal_agent = TemporalAgent(self.settings)
        self._judge_agent = JudgeAgent(self.settings)
        self._report_agent = ReportAgent(self.settings)

        self._agents_initialized = True
        self.logger.info("All agents initialized")

    async def _notify_progress(
        self, agent_name: str, progress: float, status: str
    ) -> None:
        """Send progress update via callback if available."""
        if self.progress_callback:
            try:
                await self.progress_callback(agent_name, progress, status)
            except Exception as e:
                self.logger.warning("Progress callback error: %s", e)

    async def run(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        url: Optional[str] = None,
    ) -> FactCheckState:
        """
        Execute the full fact-checking pipeline.

        Args:
            text: Input text to fact-check.
            image_path: Path to uploaded image.
            url: URL to fact-check.

        Returns:
            Completed FactCheckState with all results.
        """
        self._init_agents()

        # Create initial state
        fact_check_id = f"fc-{uuid.uuid4().hex[:12]}"
        state = FactCheckState(
            fact_check_id=fact_check_id,
            original_text=text,
            image_path=image_path,
            url=url,
            status="running",
        )

        pipeline_start = time.time()
        self.logger.info("Starting fact-check pipeline: %s", fact_check_id)

        try:
            # ── Step 1: Input Understanding ──────────────────────────
            state = await self._run_agent(
                self._input_agent, state, "Input Understanding", 0.10
            )

            # Determine pipeline based on input analysis
            input_analysis = state.input_analysis
            if input_analysis is None:
                raise ValueError("Input analysis agent returned no result")

            # ── Step 2: Image Analysis (if needed) ───────────────────
            if (
                input_analysis.requires_image_analysis
                and state.image_path
                and self.settings.enable_image_agent
            ):
                state = await self._run_agent(
                    self._image_agent, state, "Image Analysis", 0.20
                )

                # If image produced text and no original text exists,
                # use extracted text for claim extraction
                if (
                    state.image_analysis
                    and state.image_analysis.ocr_text
                    and not state.original_text
                ):
                    state.original_text = state.image_analysis.ocr_text

            # ── Step 3: Claim Extraction ─────────────────────────────
            state = await self._run_agent(
                self._claim_agent, state, "Claim Extraction", 0.30
            )

            # Check if we have verifiable claims
            factual_claims = []
            if state.claim_extraction and state.claim_extraction.claims:
                for c in state.claim_extraction.claims:
                    c_type_str = str(getattr(c.claim_type, "value", c.claim_type)).lower()
                    if c_type_str in ("factual", "claimtype.factual") or c.claim_type == ClaimType.FACTUAL:
                        factual_claims.append(c)

                # Fallback: If no claim was explicitly marked factual, treat all non-opinion claims as factual
                if not factual_claims:
                    factual_claims = [
                        c for c in state.claim_extraction.claims
                        if str(getattr(c.claim_type, "value", c.claim_type)).lower() != "opinion"
                    ]

            if not factual_claims:
                self.logger.info("No factual claims to verify")
                # Still generate report for non-factual content
                state = await self._run_agent(
                    self._judge_agent, state, "Final Judgment", 0.85
                )
                state = await self._run_agent(
                    self._report_agent, state, "Report Generation", 0.95
                )
                state.status = "completed"
                state.progress_percent = 1.0
                return state

            # ── Step 4: Web Research ─────────────────────────────────
            if self.settings.enable_web_search and (input_analysis.requires_web_search or factual_claims):
                state = await self._run_agent(
                    self._research_agent, state, "Web Research", 0.45
                )

            # ── Step 5: Evidence Verification ────────────────────────
            state = await self._run_agent(
                self._evidence_agent, state, "Evidence Verification", 0.55
            )

            # ── Step 6: Contradiction Detection ──────────────────────
            if self.settings.enable_contradiction_agent:
                state = await self._run_agent(
                    self._contradiction_agent,
                    state,
                    "Contradiction Detection",
                    0.65,
                )

            # ── Step 7: Temporal Reasoning ───────────────────────────
            if self.settings.enable_temporal_agent:
                state = await self._run_agent(
                    self._temporal_agent, state, "Temporal Analysis", 0.75
                )

            # ── Step 8: Final Judgment ───────────────────────────────
            state = await self._run_agent(
                self._judge_agent, state, "Final Judgment", 0.85
            )

            # ── Step 9: Report Generation ────────────────────────────
            state = await self._run_agent(
                self._report_agent, state, "Report Generation", 0.95
            )

            # ── Complete ─────────────────────────────────────────────
            state.status = "completed"
            state.progress_percent = 1.0
            state.total_duration_seconds = time.time() - pipeline_start

            # Sum up totals
            state.total_llm_calls = sum(
                t.llm_calls for t in state.agent_trace
            )
            state.total_web_searches = sum(
                t.web_searches for t in state.agent_trace
            )

            await self._notify_progress("Pipeline Complete", 1.0, "completed")
            self.logger.info(
                "Fact-check completed: %s (%.1fs, %d LLM calls, %d searches)",
                fact_check_id,
                state.total_duration_seconds,
                state.total_llm_calls,
                state.total_web_searches,
            )

            # Save to database
            await self._save_to_db(state)

            return state

        except Exception as e:
            state.status = "error"
            state.error_message = str(e)
            state.total_duration_seconds = time.time() - pipeline_start
            self.logger.error("Fact-check failed: %s — %s", fact_check_id, e, exc_info=True)
            await self._notify_progress("Error", state.progress_percent, "error")

            # Still try to save partial results
            try:
                await self._save_to_db(state)
            except Exception:
                self.logger.error("Failed to save error state to DB")

            return state

    async def _run_agent(
        self,
        agent: Any,
        state: FactCheckState,
        agent_display_name: str,
        progress: float,
    ) -> FactCheckState:
        """
        Execute a single agent with tracing and progress updates.

        Args:
            agent: The agent instance to execute.
            state: Current pipeline state.
            agent_display_name: Human-readable name for progress display.
            progress: Progress percentage (0.0-1.0).

        Returns:
            Updated FactCheckState.
        """
        state.current_agent = agent_display_name
        state.progress_percent = progress
        await self._notify_progress(agent_display_name, progress, "running")

        trace = AgentTraceEntry(
            agent_name=agent_display_name,
            started_at=datetime.utcnow().isoformat(),
            status="running",
        )

        start_time = time.time()
        try:
            state = await agent.execute(state)
            trace.status = "completed"
            self.logger.info("Agent completed: %s", agent_display_name)
        except Exception as e:
            trace.status = "error"
            trace.error_message = str(e)
            self.logger.error(
                "Agent failed: %s — %s", agent_display_name, e, exc_info=True
            )
            # Don't re-raise for non-critical agents; let pipeline continue
            if agent_display_name in ("Final Judgment", "Input Understanding", "Claim Extraction"):
                raise  # These are critical

        trace.completed_at = datetime.utcnow().isoformat()
        trace.duration_seconds = time.time() - start_time

        # Count LLM calls and searches from agent if tracked
        if hasattr(agent, "_llm_call_count"):
            trace.llm_calls = agent._llm_call_count
            agent._llm_call_count = 0
        if hasattr(agent, "_search_count"):
            trace.web_searches = agent._search_count
            agent._search_count = 0

        state.agent_trace.append(trace)
        return state

    async def _save_to_db(self, state: FactCheckState) -> None:
        """Persist fact-check results to SQLite database."""
        try:
            from backend.models.database import get_session_direct, FactCheckRecord, ClaimRecord

            session = await get_session_direct()
            try:
                # Create fact check record
                record = FactCheckRecord(
                    id=state.fact_check_id,
                    created_at=datetime.fromisoformat(state.created_at),
                    input_type=state.input_analysis.input_type.value if state.input_analysis else "text",
                    original_text=state.original_text,
                    input_file=state.image_path,
                    overall_verdict=(
                        state.judgment.overall_verdict.value
                        if state.judgment else None
                    ),
                    overall_confidence=(
                        state.judgment.overall_confidence
                        if state.judgment else None
                    ),
                    final_score=(
                        state.judgment.overall_score.final_score
                        if state.judgment and state.judgment.overall_score else None
                    ),
                    language=(
                        state.input_analysis.language
                        if state.input_analysis else "en"
                    ),
                    status=state.status,
                    error_message=state.error_message,
                    total_duration_seconds=state.total_duration_seconds,
                    total_llm_calls=state.total_llm_calls,
                    total_web_searches=state.total_web_searches,
                    report_json=(
                        state.report.model_dump_json()
                        if state.report else None
                    ),
                )
                session.add(record)

                # Add claim records
                if state.claim_extraction:
                    for claim in state.claim_extraction.claims:
                        # Find verdict for this claim
                        verdict_data = None
                        if state.judgment:
                            for cv in state.judgment.claim_verdicts:
                                if cv.claim_id == claim.claim_id:
                                    verdict_data = cv
                                    break

                        claim_record = ClaimRecord(
                            fact_check_id=state.fact_check_id,
                            claim_id=claim.claim_id,
                            claim_text=claim.claim,
                            claim_type=claim.claim_type.value,
                            importance=claim.importance.value,
                            verdict=verdict_data.verdict.value if verdict_data else None,
                            confidence=verdict_data.confidence if verdict_data else None,
                            reasoning=verdict_data.reasoning_summary if verdict_data else None,
                        )
                        session.add(claim_record)

                await session.commit()
                self.logger.info("Saved to database: %s", state.fact_check_id)
            except Exception as e:
                await session.rollback()
                self.logger.error("Database save failed: %s", e)
                raise
            finally:
                await session.close()

        except ImportError:
            self.logger.warning("Database not available, skipping save")
        except Exception as e:
            self.logger.error("Failed to save to database: %s", e)
