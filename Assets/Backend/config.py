"""
Central configuration for the Multimodal Multi-Agent News Fact-Checking System.

Loads settings from environment variables (.env file) with sensible defaults.
All configurable parameters are centralized here to support ablation experiments.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── OpenAI Configuration ──────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-4o",
        description="Primary model for complex reasoning tasks",
    )
    openai_mini_model: str = Field(
        default="gpt-4o-mini",
        description="Lighter model for classification / simple tasks",
    )

    # ── Optional External Search API ──────────────────────────────────
    search_api_key: Optional[str] = Field(
        default=None, description="Optional external search API key"
    )
    search_api_provider: Optional[str] = Field(
        default=None,
        description="External search provider: serpapi, google_custom, bing",
    )

    # ── Backend ───────────────────────────────────────────────────────
    backend_url: str = Field(default="http://localhost:8000")
    backend_host: str = Field(default="0.0.0.0")
    backend_port: int = Field(default=8000)

    # ── Storage Paths ─────────────────────────────────────────────────
    data_dir: str = Field(default="data")
    sqlite_db_path: str = Field(default="data/factcheck.db")

    # ── Search Settings ───────────────────────────────────────────────
    max_search_queries_per_claim: int = Field(default=5)
    max_sources_per_claim: int = Field(default=10)

    # ── Fact-Check Score Weights ──────────────────────────────────────
    # These are initial heuristics and must be configurable.
    # They are NOT scientifically validated.
    weight_evidence_strength: float = Field(default=0.30)
    weight_source_credibility: float = Field(default=0.25)
    weight_cross_source_consistency: float = Field(default=0.20)
    weight_temporal_consistency: float = Field(default=0.15)
    weight_image_context_consistency: float = Field(default=0.10)

    # ── Source Credibility Tier Scores ────────────────────────────────
    # Tier 1: Primary/official (government, regulators, court docs)
    # Tier 2: Highly reputable journalism (Reuters, AP, BBC, AFP)
    # Tier 3: Established secondary publications
    # Tier 4: Unknown blogs / aggregators
    # Tier 5: Social media / unverified posts
    tier1_score: float = Field(default=1.00)
    tier2_score: float = Field(default=0.90)
    tier3_score: float = Field(default=0.75)
    tier4_score: float = Field(default=0.50)
    tier5_score: float = Field(default=0.20)

    # ── Ablation Experiment Flags ─────────────────────────────────────
    enable_web_search: bool = Field(default=True)
    enable_contradiction_agent: bool = Field(default=True)
    enable_temporal_agent: bool = Field(default=True)
    enable_image_agent: bool = Field(default=True)
    enable_source_ranking: bool = Field(default=True)
    enable_claim_decomposition: bool = Field(default=True)

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    model_config = {
        "env_file": (".env", ".env.local"),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    # ── Derived Paths ─────────────────────────────────────────────────

    @property
    def uploads_dir(self) -> Path:
        return Path(self.data_dir) / "uploads"

    @property
    def images_dir(self) -> Path:
        return Path(self.data_dir) / "images"

    @property
    def ocr_dir(self) -> Path:
        return Path(self.data_dir) / "ocr"

    @property
    def search_results_dir(self) -> Path:
        return Path(self.data_dir) / "search_results"

    @property
    def evidence_dir(self) -> Path:
        return Path(self.data_dir) / "evidence"

    @property
    def reports_dir(self) -> Path:
        return Path(self.data_dir) / "reports"

    @property
    def logs_dir(self) -> Path:
        return Path(self.data_dir) / "logs"

    @property
    def evaluation_dir(self) -> Path:
        return Path(self.data_dir) / "evaluation"

    @property
    def cache_dir(self) -> Path:
        return Path(self.data_dir) / "cache"

    def ensure_directories(self) -> None:
        """Create all required data directories if they don't exist."""
        for dir_path in [
            self.uploads_dir,
            self.images_dir,
            self.ocr_dir,
            self.search_results_dir,
            self.evidence_dir,
            self.reports_dir,
            self.logs_dir,
            self.evaluation_dir,
            self.cache_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_tier_score(self, tier: int) -> float:
        """Get credibility score for a source tier."""
        tier_map = {
            1: self.tier1_score,
            2: self.tier2_score,
            3: self.tier3_score,
            4: self.tier4_score,
            5: self.tier5_score,
        }
        return tier_map.get(tier, self.tier5_score)

    def get_score_weights(self) -> dict[str, float]:
        """Return all score weights as a dictionary."""
        return {
            "evidence_strength": self.weight_evidence_strength,
            "source_credibility": self.weight_source_credibility,
            "cross_source_consistency": self.weight_cross_source_consistency,
            "temporal_consistency": self.weight_temporal_consistency,
            "image_context_consistency": self.weight_image_context_consistency,
        }


def get_settings() -> Settings:
    """Factory function for settings singleton."""
    return Settings()
