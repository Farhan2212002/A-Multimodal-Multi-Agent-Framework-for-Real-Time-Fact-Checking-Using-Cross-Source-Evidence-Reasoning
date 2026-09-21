import json
import time
from datetime import datetime

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import FactCheckState, ImageAnalysisResult

try:
    from backend.tools.image_tools import encode_image_to_base64
except ImportError:
    def encode_image_to_base64(image_path: str) -> str:
        import base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

try:
    from backend.tools.ocr import perform_ocr
except ImportError:
    async def perform_ocr(image_path: str) -> str:
        # Dummy fallback
        return ""

class ImageAgent(BaseAgent):
    """
    Image Analysis Agent:
    - Uses OpenAI Vision API for semantic visual reasoning
    - Runs OCR
    - Detects manipulation, context, implied claims
    """

    async def execute(self, state: FactCheckState) -> FactCheckState:
        self.logger.info(f"Starting {self.agent_name} for FactCheck ID: {state.fact_check_id}")
        trace = self._create_trace_entry()
        start_time = time.time()
        
        try:
            if not self.settings.enable_image_agent or not state.image_path:
                self.logger.info("Image analysis disabled or no image provided.")
                trace.status = "completed"
                return state

            # Perform OCR
            ocr_text = await perform_ocr(state.image_path)
            
            # Perform Semantic Analysis via Vision API
            base64_image = encode_image_to_base64(state.image_path)
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an Image Analysis Agent. Perform semantic visual reasoning on the image.\n"
                        "Consider:\n"
                        "- What objects/entities are visible?\n"
                        "- Is there visible text? What does it say?\n"
                        "- Is there a date, location, logo, person?\n"
                        "- What event does the image appear to depict?\n"
                        "- What claim does the image imply?\n"
                        "- Are there signs of potential manipulation? (obvious artifacts, inconsistencies)\n"
                        "- Could this image be presented out of context?\n"
                        "Never claim certainty about authenticity without evidence. "
                        "If you cannot verify origin, output: 'Image origin could not be independently verified.'\n"
                        "Never invent sources, URLs, or quotations. If evidence is unavailable, say so. Separate model reasoning from retrieved evidence.\n"
                        "Respond ONLY with a JSON object matching this schema:\n"
                        "{\n"
                        '  "ocr_text": "string or null",\n'
                        '  "visual_description": "string",\n'
                        '  "detected_entities": ["string"],\n'
                        '  "detected_location": "string or null",\n'
                        '  "detected_date": "string or null",\n'
                        '  "detected_source_brand": "string or null",\n'
                        '  "implied_claims": ["string"],\n'
                        '  "context_findings": ["string"],\n'
                        '  "potential_manipulation_indicators": ["string"],\n'
                        '  "image_context_verdict": "AUTHENTIC_CONTEXT" | "MISLEADING_CONTEXT" | "MANIPULATED" | "OUT_OF_CONTEXT" | "UNKNOWN",\n'
                        '  "confidence": float\n'
                        "}"
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this image. OCR previously extracted: {ocr_text}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            response_json = await self._call_llm(
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            result_dict = json.loads(response_json)
            # Ensure OCR text is included if the model missed it
            if not result_dict.get("ocr_text"):
                result_dict["ocr_text"] = ocr_text
                
            state.image_analysis = ImageAnalysisResult(**result_dict)
            
            trace.status = "completed"
            trace.llm_calls = 1
            
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
