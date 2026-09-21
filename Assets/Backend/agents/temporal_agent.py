import json
import logging
from datetime import datetime

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import (
    FactCheckState,
    TemporalAnalysis,
    TemporalStatus,
    AgentTraceEntry
)


class TemporalAgent(BaseAgent):
    """
    Temporal Reasoning Agent:
    - Analyzes temporal aspects of each claim
    - Identifies: event_time, publication_time, verification_time
    - Uses time_tools from backend.tools.time_tools for date parsing (implied by design)
    - Classifies: CURRENT, HISTORICAL, OUTDATED, FUTURE, UNCONFIRMED, TIME_INCONSISTENT
    - Key checks:
      - Is the claim about a past event being presented as current?
      - Is the article from years ago being recirculated?
      - Are source dates consistent with the claimed event time?
      - Is the claim about a future event that hasn't happened yet?
    - Computes temporal_consistency_score (1.0 = temporally consistent, lower = issues detected)
    - If claim has no meaningful temporal component, quickly mark as CURRENT with high confidence
    """

    async def execute(self, state: FactCheckState) -> FactCheckState:
        trace = self._create_trace_entry()
        self.logger.info("Executing TemporalAgent")

        try:
            state.temporal_results = []
            
            # Map claims for quick access
            claims_map = {}
            if state.claim_extraction:
                claims_map = {c.claim_id: c for c in state.claim_extraction.claims}

            for ev_result in state.evidence_results:
                claim_id = ev_result.claim_id
                claim_obj = claims_map.get(claim_id)
                claim_text = claim_obj.claim if claim_obj else ""
                claim_date = claim_obj.date if claim_obj else ""

                if not claim_obj:
                    continue
                
                # Gather publication dates from evidence assessments
                sources_data = []
                for a in ev_result.evidence_assessments:
                    # Look up publication date from web_research_results
                    pub_date = None
                    for wr in state.web_research_results:
                        if wr.claim_id == claim_id:
                            for src in wr.sources:
                                if src.url == a.source_url:
                                    pub_date = src.publication_date
                                    break
                    sources_data.append({
                        "url": a.source_url,
                        "title": a.source_title,
                        "publication_date": pub_date
                    })

                prompt = f"""You are a Temporal Reasoning Agent for a fact-checking system.
Your task is to analyze the temporal aspects of a claim and its supporting sources.
Identify the event time, publication time of sources, and current verification time.

Consider these temporal misinformation examples:
- An old article from 5 years ago is reposted as breaking news today.
- An image from a historical event is claimed to be from a protest that happened yesterday.
- A future prediction is stated as a current fact.

Claim: "{claim_text}"
Extracted Claim Date: "{claim_date}"
Sources:
{json.dumps(sources_data, indent=2)}

Analyze the claim and respond in JSON format:
{{
    "temporal_status": "CURRENT|HISTORICAL|OUTDATED|FUTURE|UNCONFIRMED|TIME_INCONSISTENT",
    "event_time": "best estimate of when the event occurred (string, ISO or description), or null",
    "publication_time": "best estimate of when the key sources were published (string) or null",
    "temporal_consistency_score": float (0.0 to 1.0, 1.0 = consistent, lower = inconsistencies),
    "is_potentially_outdated": true/false,
    "reason": "explanation of temporal analysis"
}}
If the claim has no meaningful temporal component, mark as CURRENT with temporal_consistency_score 1.0 and is_potentially_outdated false.
IMPORTANT: Do not hallucinate. Use only the provided information.
"""
                messages = [
                    {"role": "system", "content": "You are a highly capable temporal reasoning agent that outputs valid JSON."},
                    {"role": "user", "content": prompt}
                ]

                response_json = await self._call_llm_json(messages, model=self.settings.openai_model)
                trace.llm_calls += 1
                
                # Map string status to enum
                status_str = response_json.get("temporal_status", "UNCONFIRMED")
                try:
                    temporal_status = TemporalStatus(status_str)
                except ValueError:
                    temporal_status = TemporalStatus.UNCONFIRMED

                result = TemporalAnalysis(
                    claim_id=claim_id,
                    temporal_status=temporal_status,
                    event_time=response_json.get("event_time"),
                    publication_time=response_json.get("publication_time"),
                    temporal_consistency_score=float(response_json.get("temporal_consistency_score", 1.0)),
                    is_potentially_outdated=bool(response_json.get("is_potentially_outdated", False)),
                    reason=response_json.get("reason", "")
                )

                state.temporal_results.append(result)

            trace.status = "completed"
        except Exception as e:
            self.logger.error(f"Error in TemporalAgent: {str(e)}")
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
