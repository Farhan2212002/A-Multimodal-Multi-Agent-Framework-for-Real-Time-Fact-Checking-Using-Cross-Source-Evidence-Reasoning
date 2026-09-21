import json
import time
from datetime import datetime

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import FactCheckState, ClaimExtractionResult, Claim

class ClaimAgent(BaseAgent):
    """
    Claim Extraction Agent:
    - Decomposes text into atomic factual claims
    - Identifies entities, dates, locations
    - Marks opinions as NOT_FACT_CHECKABLE
    - Handles Bangla and English
    """

    async def execute(self, state: FactCheckState) -> FactCheckState:
        self.logger.info(f"Starting {self.agent_name} for FactCheck ID: {state.fact_check_id}")
        trace = self._create_trace_entry()
        start_time = time.time()
        
        try:
            # Determine text to process
            text_to_process = ""
            if state.input_analysis:
                if state.input_analysis.original_text:
                    text_to_process += state.input_analysis.original_text + "\n"
                if state.input_analysis.extracted_text_from_image:
                    text_to_process += state.input_analysis.extracted_text_from_image + "\n"
            
            if not text_to_process.strip() and state.original_text:
                text_to_process = state.original_text
                
            text_to_process = text_to_process.strip()
            
            if not text_to_process:
                self.logger.info("No text to process for claim extraction.")
                state.claim_extraction = ClaimExtractionResult()
                trace.status = "completed"
                return state
                
            # Handle ablation: disable claim decomposition
            if not self.settings.enable_claim_decomposition:
                self.logger.info("Claim decomposition disabled (ablation). Treating as single claim.")
                claim = Claim(
                    claim_id="C001",
                    claim=text_to_process,
                    claim_type="factual",
                    importance="high"
                )
                state.claim_extraction = ClaimExtractionResult(
                    claims=[claim],
                    total_claims=1,
                    factual_claims=1
                )
                trace.status = "completed"
                return state

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Claim Extraction Agent. Extract atomic, independently verifiable "
                        "factual claims from the provided text (which may be English or Bangla).\n"
                        "Rules:\n"
                        "1. Break complex sentences into simple, atomic claims.\n"
                        "2. Separate facts from opinions/predictions. Mark opinions with claim_type='opinion'.\n"
                        "3. Identify key entities, locations, dates, and numeric facts.\n"
                        "4. Assign a unique ID to each claim (C001, C002, ...).\n"
                        "5. Assign importance (high, medium, low).\n"
                        "6. Never invent sources, URLs, or quotations. If evidence is unavailable, say so. Separate model reasoning from retrieved evidence.\n\n"
                        "Respond ONLY with a JSON object matching this schema:\n"
                        "{\n"
                        '  "claims": [\n'
                        '    {\n'
                        '      "claim_id": "C001",\n'
                        '      "claim": "Atomic claim text",\n'
                        '      "claim_type": "factual" | "opinion" | "prediction",\n'
                        '      "importance": "high" | "medium" | "low",\n'
                        '      "entities": ["Entity1", "Entity2"],\n'
                        '      "location": "Location or null",\n'
                        '      "date": "Date or null",\n'
                        '      "numeric_facts": ["1000 people", "50%"]\n'
                        '    }\n'
                        '  ],\n'
                        '  "total_claims": 1,\n'
                        '  "factual_claims": 1,\n'
                        '  "opinion_claims": 0,\n'
                        '  "prediction_claims": 0\n'
                        "}\n\n"
                        "Example:\n"
                        "Text: 'The mayor announced on Monday that crime dropped by 50% in New York. However, I think he is a terrible mayor.'\n"
                        "JSON: ... extracts C001 (mayor announced crime dropped by 50% in NY on Monday) as factual, "
                        "and C002 (mayor is terrible) as opinion."
                    )
                },
                {
                    "role": "user",
                    "content": f"Extract claims from the following text:\n\n{text_to_process}"
                }
            ]
            
            response_json = await self._call_llm(
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            result_dict = json.loads(response_json)
            state.claim_extraction = ClaimExtractionResult(**result_dict)
            
            trace.status = "completed"
            trace.llm_calls = 1
            
        except Exception as e:
            self.logger.error(f"Error in {self.agent_name}: {str(e)}")
            trace.status = "error"
            trace.error_message = str(e)
            # Create a heuristic claim fallback so research and evidence agents can proceed
            claim = Claim(
                claim_id="C001",
                claim=text_to_process,
                claim_type="factual",
                importance="high"
            )
            state.claim_extraction = ClaimExtractionResult(
                claims=[claim],
                total_claims=1,
                factual_claims=1
            )
            
        finally:
            trace.completed_at = datetime.utcnow().isoformat()
            trace.duration_seconds = time.time() - start_time
            state.agent_trace.append(trace)
            state.total_llm_calls += trace.llm_calls
            state.total_duration_seconds += trace.duration_seconds
            
        return state
