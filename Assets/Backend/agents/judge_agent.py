"""
Judge Agent — evidence-grounded final decision agent.

Receives all structured evidence from prior agents via FactCheckState.
Evaluates evidence support, contradictions, source credibility, independence,
temporal consistency, and image context consistency to assign verdicts.
"""

import json
import logging
from datetime import datetime

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import (
    AgentTraceEntry,
    ClaimVerdict,
    FactCheckScore,
    FactCheckState,
    JudgmentResult,
    Verdict,
)

logger = logging.getLogger(__name__)


class JudgeAgent(BaseAgent):
    """
    Judge Agent:
    - Receives ALL structured evidence from prior agents via FactCheckState
    - Considers: evidence support, contradiction, source credibility, source independence,
      temporal consistency, image/context consistency, claim specificity
    - Computes FactCheckScore using configurable weights from settings
    - Maps score to verdict
    - CRITICAL: The Judge must NEVER invent evidence. Only cite evidence from state.
    - CRITICAL: If evidence is insufficient, return UNVERIFIABLE
    - Produces per-claim ClaimVerdict and overall JudgmentResult
    """

    async def execute(self, state: FactCheckState) -> FactCheckState:
        trace = self._create_trace_entry()
        self.logger.info("Executing JudgeAgent")

        try:
            claim_verdicts: list[ClaimVerdict] = []

            if not state.claim_extraction or not state.claim_extraction.claims:
                state.judgment = JudgmentResult(
                    claim_verdicts=[],
                    overall_verdict=Verdict.UNVERIFIABLE,
                    overall_confidence=0.0,
                    reasoning="No factual claims extracted to verify.",
                )
                trace.status = "completed"
                return state

            # Map evidence, contradiction, and temporal results by claim_id
            ev_map = {r.claim_id: r for r in state.evidence_results}
            contra_map = {r.claim_id: r for r in state.contradiction_results}
            temp_map = {r.claim_id: r for r in state.temporal_results}

            for claim_obj in state.claim_extraction.claims:
                claim_id = claim_obj.claim_id
                claim_text = claim_obj.claim
                c_type = str(getattr(claim_obj.claim_type, "value", claim_obj.claim_type)).lower()

                # If claim is opinion, mark NOT_FACT_CHECKABLE
                if c_type == "opinion":
                    claim_verdicts.append(
                        ClaimVerdict(
                            claim_id=claim_id,
                            claim_text=claim_text,
                            verdict=Verdict.NOT_FACT_CHECKABLE,
                            confidence=1.0,
                            reasoning_summary="This claim is an opinion and cannot be objectively fact-checked.",
                        )
                    )
                    continue

                ev_result = ev_map.get(claim_id)
                contradiction_res = contra_map.get(claim_id)
                temporal_res = temp_map.get(claim_id)

                evidence_assessments = (
                    ev_result.evidence_assessments if ev_result else []
                )
                evidence_strength = (
                    ev_result.overall_evidence_strength if ev_result else 0.0
                )
                independent_source_count = (
                    ev_result.independent_source_count if ev_result else 0
                )

                # Cross-source consistency
                cross_source_consistency = 1.0
                if contradiction_res:
                    cross_source_consistency = (
                        contradiction_res.cross_source_consistency_score
                    )

                # Temporal consistency
                temporal_consistency = 1.0
                if temporal_res:
                    temporal_consistency = (
                        temporal_res.temporal_consistency_score
                    )

                # Image context consistency
                image_context_consistency = 1.0
                if state.image_analysis:
                    img_v = state.image_analysis.image_context_verdict.value
                    if img_v == "AUTHENTIC_CONTEXT":
                        image_context_consistency = 1.0
                    elif img_v in ("MISLEADING_CONTEXT", "OUT_OF_CONTEXT"):
                        image_context_consistency = 0.5
                    elif img_v == "MANIPULATED":
                        image_context_consistency = 0.0

                # Source Credibility Score
                source_credibility = 0.5
                if evidence_assessments:
                    nodes = state.evidence_graph.get_nodes_for_claim(claim_id)
                    credibilities = [n.credibility_score for n in nodes]
                    if credibilities:
                        source_credibility = sum(credibilities) / len(
                            credibilities
                        )

                weights = self.settings.get_score_weights()

                # Calculate Composite FCS
                fcs_val = (
                    weights.get("evidence_strength", 0.30) * evidence_strength
                    + weights.get("source_credibility", 0.25) * source_credibility
                    + weights.get("cross_source_consistency", 0.20)
                    * cross_source_consistency
                    + weights.get("temporal_consistency", 0.15)
                    * temporal_consistency
                    + weights.get("image_context_consistency", 0.10)
                    * image_context_consistency
                )

                # Determine verdict from evidence and score
                supporting_nodes = (
                    state.evidence_graph.get_supporting(claim_id)
                    if state.evidence_graph
                    else []
                )
                contradicting_nodes = (
                    state.evidence_graph.get_contradicting(claim_id)
                    if state.evidence_graph
                    else []
                )

                verdict = Verdict.UNVERIFIABLE
                confidence = 0.0

                if not evidence_assessments or independent_source_count == 0:
                    verdict = Verdict.UNVERIFIABLE
                    confidence = 0.0
                elif contradicting_nodes and not supporting_nodes:
                    verdict = Verdict.FALSE
                    confidence = min(1.0, 0.5 + (len(contradicting_nodes) * 0.15))
                elif contradicting_nodes and supporting_nodes:
                    verdict = Verdict.MISLEADING
                    confidence = 0.70
                else:
                    if fcs_val >= 0.80:
                        verdict = Verdict.TRUE
                    elif fcs_val >= 0.65:
                        verdict = Verdict.MOSTLY_TRUE
                    elif fcs_val >= 0.50:
                        verdict = Verdict.PARTIALLY_TRUE
                    elif fcs_val >= 0.35:
                        verdict = Verdict.MISLEADING
                    elif fcs_val >= 0.20:
                        verdict = Verdict.MOSTLY_FALSE
                    else:
                        verdict = Verdict.FALSE
                    confidence = max(0.40, min(1.0, fcs_val))

                fc_score = FactCheckScore(
                    evidence_strength=evidence_strength,
                    source_credibility=source_credibility,
                    cross_source_consistency=cross_source_consistency,
                    temporal_consistency=temporal_consistency,
                    image_context_consistency=image_context_consistency,
                    final_score=fcs_val,
                    weights_used=weights,
                )

                # Prepare evidence summary for LLM reasoning prompt
                evidence_text = json.dumps(
                    [
                        {
                            "url": a.source_url,
                            "title": a.source_title,
                            "relation": a.relation.value,
                            "reason": a.reason,
                        }
                        for a in evidence_assessments
                    ],
                    indent=2,
                )

                prompt = f"""You are the final Judge Agent for a fact-checking system.
Review the following claim and its retrieved web evidence to provide a concise reasoning summary.
DO NOT hallucinate or invent evidence. Only cite evidence provided below.
If evidence is insufficient, explicitly state that.

Claim: "{claim_text}"
Assigned Verdict: {verdict.value} (Score: {fcs_val:.2f})
Retrieved Evidence:
{evidence_text}

Respond ONLY with a JSON object matching this schema:
{{
    "reasoning_summary": "Concise summary explaining the verdict based strictly on retrieved evidence",
    "supporting_evidence": ["list of strings summarizing key supporting points from evidence"],
    "contradicting_evidence": ["list of strings summarizing key contradicting points from evidence"],
    "limitations": ["list of strings noting limitations, e.g. unverified sources, incomplete reports"]
}}
"""
                messages = [
                    {
                        "role": "system",
                        "content": "You are a strict, evidence-grounded judge agent that outputs valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ]

                try:
                    response_json = await self._call_llm_json(
                        messages, model=self.settings.openai_model
                    )
                    trace.llm_calls += 1
                    reasoning_summary = response_json.get(
                        "reasoning_summary",
                        f"Verdict: {verdict.value} based on evidence evaluation.",
                    )
                    supporting_ev = response_json.get("supporting_evidence", [])
                    contradicting_ev = response_json.get("contradicting_evidence", [])
                    limitations = response_json.get("limitations", [])
                except Exception as e:
                    self.logger.warning(
                        f"LLM reasoning generation failed for claim {claim_id}: {e}"
                    )
                    reasoning_summary = (
                        f"Verdict: {verdict.value} based on {len(evidence_assessments)} sources."
                    )
                    supporting_ev = [
                        a.reason
                        for a in evidence_assessments
                        if a.relation.value in ("SUPPORTS", "PARTIALLY_SUPPORTS")
                    ]
                    contradicting_ev = [
                        a.reason
                        for a in evidence_assessments
                        if a.relation.value == "CONTRADICTS"
                    ]
                    limitations = ["LLM synthesis fell back to direct evidence summary."]

                claim_verdicts.append(
                    ClaimVerdict(
                        claim_id=claim_id,
                        claim_text=claim_text,
                        verdict=verdict,
                        confidence=confidence,
                        fact_check_score=fc_score,
                        reasoning_summary=reasoning_summary,
                        supporting_evidence=supporting_ev,
                        contradicting_evidence=contradicting_ev,
                        limitations=limitations,
                    )
                )

            # Overall verdict logic
            overall_verdict = Verdict.UNVERIFIABLE
            overall_confidence = 0.0
            overall_score = None
            reasoning = ""

            if claim_verdicts:
                severity = {
                    Verdict.FALSE: 6,
                    Verdict.MOSTLY_FALSE: 5,
                    Verdict.MISLEADING: 4,
                    Verdict.PARTIALLY_TRUE: 3,
                    Verdict.MOSTLY_TRUE: 2,
                    Verdict.TRUE: 1,
                    Verdict.UNVERIFIABLE: 0,
                    Verdict.NOT_FACT_CHECKABLE: -1,
                }

                best_claim = max(
                    claim_verdicts, key=lambda c: severity.get(c.verdict, 0)
                )
                overall_verdict = best_claim.verdict

                confidences = [c.confidence for c in claim_verdicts]
                overall_confidence = sum(confidences) / len(confidences)

                if best_claim.fact_check_score:
                    overall_score = best_claim.fact_check_score

                reasoning = (
                    f"Overall verdict is {overall_verdict.value} derived from individual claim verdicts."
                )

            state.judgment = JudgmentResult(
                claim_verdicts=claim_verdicts,
                overall_verdict=overall_verdict,
                overall_confidence=overall_confidence,
                overall_score=overall_score,
                reasoning=reasoning,
            )

            trace.status = "completed"
        except Exception as e:
            self.logger.error(f"Error in JudgeAgent: {str(e)}")
            trace.status = "error"
            trace.error_message = str(e)
            state.error_message = str(e)
            state.status = "error"

        finally:
            trace.completed_at = datetime.utcnow().isoformat()
            if trace.started_at:
                try:
                    start = datetime.fromisoformat(trace.started_at)
                    trace.duration_seconds = (
                        datetime.utcnow() - start
                    ).total_seconds()
                except ValueError:
                    pass

            state.agent_trace.append(trace)

        return state
