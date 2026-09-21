import json
from datetime import datetime
import time

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import FactCheckState, InputAnalysis, InputType

# Attempt to import image_tools if available, otherwise mock it
try:
    from backend.tools.image_tools import encode_image_to_base64
except ImportError:
    def encode_image_to_base64(image_path: str) -> str:
        import base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


class InputAgent(BaseAgent):
    """
    Input Understanding Agent:
    - Detects input type (text, image, mixed, url)
    - Uses OpenAI Vision API for images
    - Detects language
    - Determines processing requirements
    """

    async def execute(self, state: FactCheckState) -> FactCheckState:
        self.logger.info(f"Starting {self.agent_name} for FactCheck ID: {state.fact_check_id}")
        trace = self._create_trace_entry()
        start_time = time.time()
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an Input Understanding Agent. Analyze the user's input "
                        "and determine its type, language, and processing requirements. "
                        "Never invent sources, URLs, or quotations. If evidence is unavailable, say so. "
                        "Separate model reasoning from retrieved evidence.\n"
                        "Respond ONLY with a JSON object matching this schema:\n"
                        "{\n"
                        '  "input_type": "text" | "image" | "mixed" | "url",\n'
                        '  "language": "en" (or other ISO 639-1 code),\n'
                        '  "contains_text": boolean,\n'
                        '  "contains_visual_information": boolean,\n'
                        '  "requires_ocr": boolean,\n'
                        '  "requires_web_search": boolean,\n'
                        '  "requires_image_analysis": boolean,\n'
                        '  "original_text": "string or null",\n'
                        '  "extracted_text_from_image": "string or null",\n'
                        '  "url": "string or null",\n'
                        '  "reason": "explanation of your analysis"\n'
                        "}"
                    )
                }
            ]
            
            content = []
            
            if state.original_text:
                content.append({"type": "text", "text": f"User Text Input: {state.original_text}"})
                
            if state.url:
                content.append({"type": "text", "text": f"User URL Input: {state.url}"})
                
            if state.image_path:
                try:
                    base64_image = encode_image_to_base64(state.image_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                    content.append({"type": "text", "text": "Analyze what's in the image and whether it contains text."})
                except Exception as e:
                    self.logger.error(f"Failed to process image: {e}")
                    
            messages.append({
                "role": "user",
                "content": content
            })
            
            # Use mini model for text-only, standard model for images
            model = self.settings.openai_model if state.image_path else self.settings.openai_mini_model
            
            response_json = await self._call_llm(
                messages=messages,
                model=model,
                response_format={"type": "json_object"}
            )
            
            result_dict = json.loads(response_json)
            
            analysis = InputAnalysis(**result_dict)
            if state.original_text or state.url or analysis.contains_text:
                analysis.requires_web_search = True
            state.input_analysis = analysis
            
            trace.status = "completed"
            trace.llm_calls = 1
            
        except Exception as e:
            self.logger.error(f"Error in {self.agent_name}: {str(e)}")
            trace.status = "error"
            trace.error_message = str(e)
            state.error_message = str(e)
            
            # Fallback heuristic analysis so pipeline can proceed or report clear error downstream
            in_type = InputType.TEXT
            if state.image_path and state.original_text:
                in_type = InputType.MIXED
            elif state.image_path:
                in_type = InputType.IMAGE
            elif state.url:
                in_type = InputType.URL

            state.input_analysis = InputAnalysis(
                input_type=in_type,
                language="en",
                contains_text=bool(state.original_text),
                contains_visual_information=bool(state.image_path),
                requires_ocr=bool(state.image_path),
                requires_web_search=True,
                requires_image_analysis=bool(state.image_path),
                original_text=state.original_text,
                url=state.url,
                reason=f"Heuristic fallback analysis (LLM notice: {str(e)})"
            )
            
        finally:
            trace.completed_at = datetime.utcnow().isoformat()
            trace.duration_seconds = time.time() - start_time
            state.agent_trace.append(trace)
            state.total_llm_calls += trace.llm_calls
            state.total_duration_seconds += trace.duration_seconds
            
        return state
