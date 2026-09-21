import json
import logging
from datetime import datetime

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import (
    FactCheckState,
    FactCheckReport,
    AgentTraceEntry
)


class ReportAgent(BaseAgent):
    """
    Report Generation Agent:
    - Generates FactCheckReport from the final FactCheckState
    - Generates a human-readable summary using OpenAI (mini model)
    - Supports English and Bangla output
    - Assembles all sections into the final report
    """

    async def execute(self, state: FactCheckState) -> FactCheckState:
        trace = self._create_trace_entry()
        self.logger.info("Executing ReportAgent")

        try:
            language = state.input_analysis.language if state.input_analysis else "en"
            
            # Extract basic data
            judgment = state.judgment
            overall_verdict = judgment.overall_verdict if judgment else None
            overall_confidence = judgment.overall_confidence if judgment else 0.0
            
            # Build text for summary generation
            claims_text = ""
            if judgment and judgment.claim_verdicts:
                for cv in judgment.claim_verdicts:
                    claims_text += f"- Claim: {cv.claim_text}\n  Verdict: {cv.verdict.value}\n  Reasoning: {cv.reasoning_summary}\n"

            prompt = f"""You are a Fact-Check Report Generator.
Summarize the following fact-check results into a concise 2-3 sentence summary.
Language requirement: Generate the summary in the requested language (language code: {language}).
If the language is 'bn' (Bangla), output the summary in natural-sounding Bengali.

Input text/context: {state.original_text or "No original text provided."}
Overall Verdict: {overall_verdict.value if overall_verdict else "UNVERIFIABLE"}
Claims:
{claims_text}

Respond in JSON format:
{{
    "summary": "Your 2-3 sentence summary in the correct language"
}}
"""
            messages = [
                {"role": "system", "content": "You are a professional fact-checker generating report summaries."},
                {"role": "user", "content": prompt}
            ]

            response_json = await self._call_llm_json(messages, model=self.settings.openai_mini_model)
            trace.llm_calls += 1
            
            summary = response_json.get("summary", "")

            # Flatten sources and evidence
            all_sources = []
            for wr in state.web_research_results:
                all_sources.extend(wr.sources)
                
            all_evidence = []
            for ev in state.evidence_results:
                all_evidence.extend(ev.evidence_assessments)
                
            all_contradictions = []
            for c in state.contradiction_results:
                all_contradictions.extend(c.conflicts)

            report = FactCheckReport(
                fact_check_id=state.fact_check_id,
                input_type=state.input_analysis.input_type if state.input_analysis else "text",
                original_text=state.original_text,
                image_path=state.image_path,
                url=state.url,
                language=language,
                overall_verdict=overall_verdict,
                overall_confidence=overall_confidence,
                overall_score=judgment.overall_score if judgment else None,
                summary=summary,
                claims=state.claim_extraction.claims if state.claim_extraction else [],
                claim_verdicts=judgment.claim_verdicts if judgment else [],
                sources=all_sources,
                evidence_assessments=all_evidence,
                contradictions=all_contradictions,
                temporal_analyses=state.temporal_results,
                image_analysis=state.image_analysis,
                limitations=[],
                agent_trace=state.agent_trace
            )
            
            # Combine limitations from claims
            if judgment and judgment.claim_verdicts:
                for cv in judgment.claim_verdicts:
                    report.limitations.extend(cv.limitations)
            # deduplicate limitations
            report.limitations = list(set(report.limitations))

            state.report = report

            trace.status = "completed"
        except Exception as e:
            self.logger.error(f"Error in ReportAgent: {str(e)}")
            trace.status = "error"
            trace.error_message = str(e)

        trace.completed_at = datetime.utcnow().isoformat()
        if trace.started_at:
            try:
                start = datetime.fromisoformat(trace.started_at)
                trace.duration_seconds = (datetime.utcnow() - start).total_seconds()
            except ValueError:
                pass

        state.agent_trace.append(trace)
        return state
