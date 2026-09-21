"""
Web Research Agent.

Generates multi-angle search queries per claim, retrieves live web evidence,
scores source credibility, and populates the evidence state.
"""

import json
import logging
import time
from datetime import datetime

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import FactCheckState, SearchSource, WebResearchResult
from backend.tools.source_ranker import get_credibility_score, get_source_tier
from backend.tools.web_search import search_web

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Web Research Agent:
    - Generates search queries for each factual claim
    - Executes web searches
    - Uses source ranker to assign tiers and credibility
    - Deduplicates sources by URL
    """

    async def execute(self, state: FactCheckState) -> FactCheckState:
        self.logger.info(
            f"Starting {self.agent_name} for FactCheck ID: {state.fact_check_id}"
        )
        trace = self._create_trace_entry()
        start_time = time.time()

        state.web_research_results = []

        try:
            if not state.claim_extraction or not state.claim_extraction.claims:
                self.logger.info("No claims to research.")
                trace.status = "completed"
                return state

            trace.web_searches = 0
            all_state_sources = []
            seen_global_urls = set()

            for claim in state.claim_extraction.claims:
                c_type = str(getattr(claim.claim_type, "value", claim.claim_type)).lower()
                if c_type == "opinion":
                    continue

                research_result = WebResearchResult(claim_id=claim.claim_id)

                try:
                    # Generate queries
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a Research Query Generator. Create optimal web search queries "
                                "to verify or debunk the given claim.\n"
                                "Include queries for supporting evidence, contradictory evidence, and official sources.\n"
                                f"Generate up to {self.settings.max_search_queries_per_claim} queries.\n"
                                "If the claim relates to Bangladesh or might be in Bangla, include both English and Bangla queries.\n"
                                "Never invent sources, URLs, or quotations. If evidence is unavailable, say so. Separate model reasoning from retrieved evidence.\n"
                                "Respond ONLY with a JSON object matching this schema:\n"
                                "{\n"
                                '  "queries": ["query1", "query2"]\n'
                                "}"
                            ),
                        },
                        {"role": "user", "content": f"Claim: {claim.claim}"},
                    ]

                    response_json = await self._call_llm(
                        messages=messages,
                        response_format={"type": "json_object"},
                    )
                    trace.llm_calls += 1

                    queries = json.loads(response_json).get("queries", [])
                    if not queries:
                        queries = [claim.claim]
                    research_result.queries_used = queries

                    seen_urls = set()

                    for query in queries:
                        search_results = await search_web(
                            query, max_results=self.settings.max_sources_per_claim
                        )
                        trace.web_searches += 1

                        for sr in search_results:
                            url = sr.url
                            if not url or url in seen_urls:
                                continue
                            seen_urls.add(url)

                            tier = get_source_tier(url)
                            cred_score = get_credibility_score(url)

                            source = SearchSource(
                                title=sr.title,
                                url=url,
                                publisher=sr.publisher,
                                publication_date=sr.publication_date,
                                snippet=sr.snippet,
                                relevant_evidence=sr.snippet,
                                source_tier=tier,
                                credibility_score=cred_score,
                            )
                            research_result.sources.append(source)

                            if url not in seen_global_urls:
                                seen_global_urls.add(url)
                                all_state_sources.append(source)

                    research_result.search_successful = True

                except Exception as e:
                    self.logger.error(
                        f"Search failed for claim {claim.claim_id}: {e}"
                    )
                    research_result.search_successful = False
                    research_result.error_message = str(e)

                state.web_research_results.append(research_result)

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
            state.total_web_searches += trace.web_searches
            state.total_duration_seconds += trace.duration_seconds

        return state
