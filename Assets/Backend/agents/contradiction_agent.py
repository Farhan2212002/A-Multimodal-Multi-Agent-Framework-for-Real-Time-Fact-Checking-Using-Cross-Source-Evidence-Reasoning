import json
import logging
from datetime import datetime
from typing import Dict, Any

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import (
    FactCheckState,
    ContradictionResult,
    Contradiction,
    AgentTraceEntry
)


class ContradictionAgent(BaseAgent):
    """
    Contradiction Detection Agent:
    - Compares evidence from multiple independent sources for each claim
    - Detects: direct contradiction, partial contradiction, different dates, different locations, different definitions, old vs new info, rumor vs official
    - Does NOT blindly select the majority view
    - Reasons about WHY sources disagree
    - Computes cross_source_consistency_score (1.0 = all agree, 0.0 = complete disagreement)
    """

    async def execute(self, state: FactCheckState) -> FactCheckState:
        trace = self._create_trace_entry()
        self.logger.info("Executing ContradictionAgent")

        try:
            state.contradiction_results = []

            for ev_result in state.evidence_results:
                claim_id = ev_result.claim_id

                # find claim text
                claim_text = ""
                if state.claim_extraction:
                    for c in state.claim_extraction.claims:
                        if c.claim_id == claim_id:
                            claim_text = c.claim
                            break

                assessments = ev_result.evidence_assessments
                if len(assessments) < 2:
                    # Not enough sources to find contradictions
                    res = ContradictionResult(
                        claim_id=claim_id,
                        contradiction_detected=False,
                        conflicts=[],
                        cross_source_consistency_score=1.0
                    )
                    state.contradiction_results.append(res)
                    continue

                evidence_data = [
                    {
                        "url": a.source_url,
                        "title": a.source_title,
                        "relation": a.relation.value,
                        "strength": a.strength,
                        "reason": a.reason,
                        "is_independent": a.is_independent_source
                    }
                    for a in assessments
                ]

                prompt = f"""You are a Contradiction Detection Agent for a fact-checking system.
Your task is to compare evidence from multiple sources for a specific claim and detect any contradictions.
Do not blindly select the majority view. Consider source authority, publication time, evidence directness, source independence, claim scope, event date, and context.
Reason about WHY sources disagree.

Claim: "{claim_text}"
Evidence Assessments:
{json.dumps(evidence_data, indent=2)}

Analyze the evidence and respond in JSON format with the following structure:
{{
    "contradiction_detected": true/false,
    "cross_source_consistency_score": float (0.0 to 1.0, 1.0 = all agree, 0.0 = complete disagreement),
    "conflicts": [
        {{
            "source_a_url": "url_string",
            "source_a_title": "title_string",
            "source_b_url": "url_string",
            "source_b_title": "title_string",
            "conflict": "description of the conflict",
            "resolution": "reasoning about why they disagree and potential resolution based on source authority/time/etc.",
            "conflict_type": "type of conflict (e.g. direct contradiction, partial contradiction, different dates, different locations, different definitions, old vs new info, rumor vs official)"
        }}
    ]
}}
IMPORTANT: Do not hallucinate. Use only the provided evidence.
"""
                messages = [
                    {"role": "system", "content": "You are a precise, analytical fact-checking assistant that outputs valid JSON."},
                    {"role": "user", "content": prompt}
                ]

                response_json = await self._call_llm_json(messages, model=self.settings.openai_model)
                trace.llm_calls += 1

                conflicts = []
                for c in response_json.get("conflicts", []):
                    conflicts.append(Contradiction(
                        source_a_url=c.get("source_a_url", ""),
                        source_a_title=c.get("source_a_title", ""),
                        source_b_url=c.get("source_b_url", ""),
                        source_b_title=c.get("source_b_title", ""),
                        conflict=c.get("conflict", ""),
                        resolution=c.get("resolution", ""),
                        conflict_type=c.get("conflict_type", "")
                    ))

                result = ContradictionResult(
                    claim_id=claim_id,
                    contradiction_detected=response_json.get("contradiction_detected", False),
                    cross_source_consistency_score=float(response_json.get("cross_source_consistency_score", 1.0)),
                    conflicts=conflicts
                )

                state.contradiction_results.append(result)

            trace.status = "completed"
        except Exception as e:
            self.logger.error(f"Error in ContradictionAgent: {str(e)}")
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
