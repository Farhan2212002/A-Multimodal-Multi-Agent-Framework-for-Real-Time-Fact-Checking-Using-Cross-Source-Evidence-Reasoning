import json
import time
from datetime import datetime

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import FactCheckState, EvidenceVerificationResult, EvidenceAssessment, EvidenceNode

class EvidenceAgent(BaseAgent):
    """
    Evidence Verification Agent:
    - Classifies evidence relationship (SUPPORTS, CONTRADICTS, etc.)
    - Assigns strength score
    - Identifies source independence
    - Builds EvidenceNode entries for the evidence graph
    """

    async def execute(self, state: FactCheckState) -> FactCheckState:
        self.logger.info(f"Starting {self.agent_name} for FactCheck ID: {state.fact_check_id}")
        trace = self._create_trace_entry()
        start_time = time.time()
        
        state.evidence_results = []
        
        try:
            if not state.claim_extraction or not state.claim_extraction.claims:
                trace.status = "completed"
                return state

            # Map research results by claim_id for quick access
            research_map = {r.claim_id: r for r in state.web_research_results}

            for claim in state.claim_extraction.claims:
                c_type = str(getattr(claim.claim_type, "value", claim.claim_type)).lower()
                if c_type == "opinion":
                    continue
                    
                research = research_map.get(claim.claim_id)
                if not research or not research.sources:
                    continue

                # Prepare sources for batch processing
                sources_data = []
                for i, src in enumerate(research.sources):
                    sources_data.append({
                        "id": i,
                        "url": src.url,
                        "title": src.title,
                        "snippet": src.snippet
                    })

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an Evidence Verification Agent. Evaluate the provided sources against the claim.\n"
                            "For each source, determine:\n"
                            "- Relation: SUPPORTS, PARTIALLY_SUPPORTS, CONTRADICTS, IRRELEVANT, INSUFFICIENT\n"
                            "- Strength: 0.0 to 1.0 (based on directness, specificity, authority)\n"
                            "- Independence: Are these sources independent or just republishing the same content?\n"
                            "Never invent sources, URLs, or quotations. If evidence is unavailable, say so. Separate model reasoning from retrieved evidence.\n"
                            "Respond ONLY with a JSON object matching this schema:\n"
                            "{\n"
                            '  "assessments": [\n'
                            '    {\n'
                            '      "source_id": integer,\n'
                            '      "relation": "SUPPORTS",\n'
                            '      "strength": 0.8,\n'
                            '      "reason": "explanation",\n'
                            '      "is_independent_source": boolean,\n'
                            '      "possible_common_origin": "string or null"\n'
                            '    }\n'
                            '  ],\n'
                            '  "overall_evidence_strength": 0.8,\n'
                            '  "independent_source_count": 2\n'
                            "}"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Claim: {claim.claim}\nSources: {json.dumps(sources_data, indent=2)}"
                    }
                ]
                
                response_json = await self._call_llm(
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                trace.llm_calls += 1
                
                result_dict = json.loads(response_json)
                
                evidence_result = EvidenceVerificationResult(claim_id=claim.claim_id)
                evidence_result.overall_evidence_strength = result_dict.get("overall_evidence_strength", 0.0)
                evidence_result.independent_source_count = result_dict.get("independent_source_count", 0)
                
                assessments_data = result_dict.get("assessments", [])
                
                for assessment_data in assessments_data:
                    src_idx = assessment_data.get("source_id")
                    if src_idx is None or src_idx < 0 or src_idx >= len(research.sources):
                        continue
                        
                    src = research.sources[src_idx]
                    
                    assessment = EvidenceAssessment(
                        source_url=src.url,
                        source_title=src.title,
                        relation=assessment_data.get("relation", "INSUFFICIENT"),
                        strength=assessment_data.get("strength", 0.0),
                        reason=assessment_data.get("reason", ""),
                        is_independent_source=assessment_data.get("is_independent_source", True),
                        possible_common_origin=assessment_data.get("possible_common_origin")
                    )
                    evidence_result.evidence_assessments.append(assessment)
                    
                    # Update Evidence Graph
                    node = EvidenceNode(
                        claim_id=claim.claim_id,
                        source_url=src.url,
                        source_title=src.title,
                        source_tier=src.source_tier,
                        credibility_score=src.credibility_score,
                        relation=assessment.relation,
                        strength=assessment.strength,
                        is_independent=assessment.is_independent_source,
                        snippet=src.snippet
                    )
                    state.evidence_graph.nodes.append(node)
                    
                state.evidence_results.append(evidence_result)

            trace.status = "completed"
            
        except Exception as e:
            self.logger.error(f"Error in {self.agent_name}: {str(e)}")
            trace.status = "error"
            trace.error_message = str(e)
            state.error_message = str(e)
            state.status = "error"
            
        finally:
            trace.completed_at = datetime.utcnow().isoformat()
            trace.duration_seconds = time.time() - start_time
            state.agent_trace.append(trace)
            state.total_llm_calls += trace.llm_calls
            state.total_duration_seconds += trace.duration_seconds
            
        return state
