import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import openai
from pydantic import BaseModel
import json

from backend.config import Settings
from backend.models.schemas import FactCheckState, AgentTraceEntry


class BaseAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        api_key = settings.openai_api_key or "missing-key"
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.agent_name = self.__class__.__name__
        self.logger = logging.getLogger(self.agent_name)
        self._llm_call_count = 0
        self._search_count = 0
        
        # Configure logging if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    async def execute(self, state: FactCheckState) -> FactCheckState:
        """Execute agent logic. Must be overridden."""
        raise NotImplementedError("Each agent must implement the execute method.")

    async def _call_llm(
        self, 
        messages: List[Dict[str, Any]], 
        model: Optional[str] = None, 
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """Call OpenAI chat completions with error handling."""
        model = model or self.settings.openai_model
        
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
            }
            if response_format:
                kwargs["response_format"] = response_format

            self.logger.debug(f"Calling LLM ({model})")
            response = await self.client.chat.completions.create(**kwargs)
            self._llm_call_count += 1
            return response.choices[0].message.content or ""
        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            raise

    async def _call_llm_json(
        self, 
        messages: List[Dict[str, Any]], 
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Call LLM and parse JSON response."""
        response_str = await self._call_llm(
            messages=messages, 
            model=model, 
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {response_str}")
            raise ValueError(f"Invalid JSON from LLM: {str(e)}")

    def _create_trace_entry(self) -> AgentTraceEntry:
        """Create a trace entry for this agent execution."""
        return AgentTraceEntry(
            agent_name=self.agent_name,
            started_at=datetime.utcnow().isoformat(),
            status="running"
        )
