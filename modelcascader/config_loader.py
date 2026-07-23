"""
config_loader.py
Load and validate cascade_config.yaml using Pydantic v2 models.

All routing knobs flow through these typed models, so a misconfigured YAML
produces a clear validation error at startup rather than a cryptic KeyError
inside the hot path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class RoutingConfig(BaseModel):
    """General routing behaviour (timeouts, fail-safe policy)."""

    timeout_seconds: float = Field(10.0, gt=0, description="Router call timeout in seconds.")
    fail_safe: Literal["escalate"] = "escalate"
    """
    On Controller error/timeout: 'escalate' means go UP to the next tier.
    Silently routing DOWN to a cheaper model on failure is explicitly prohibited.
    """


class GatekeeperConfig(BaseModel):
    """Configuration for one binary RouteLLM gatekeeper."""

    router: Literal["mf", "bert", "causal_llm", "matrix_factorization", "sw_ranking", "random"] = "mf"
    threshold: float = Field(..., ge=0.0, le=1.0, description=(
        "Win-probability cutoff [0, 1]. Scores BELOW this route to the weaker tier; "
        "scores AT OR ABOVE escalate to the stronger tier. "
        "Produced by `python -m routellm.calibrate_threshold` — do not invent by hand."
    ))
    weak_model: str = Field(..., description="Weak-side model ID passed to RouteLLM Controller.")
    strong_model: str = Field(..., description="Strong-side model ID passed to RouteLLM Controller.")


class TierConfig(BaseModel):
    """Backend model to call for a given tier."""

    provider: Literal["openai", "anthropic", "groq", "google"] = Field(..., description="LLM provider.")
    model: str = Field(..., description="Provider-specific model identifier (e.g. 'gpt-4o').")
    label: str = Field(..., description="Human-readable tier name for logging (e.g. 'Small').")


class TiersConfig(BaseModel):
    """All three tier definitions."""

    tier_1: TierConfig
    tier_2: TierConfig
    tier_3: TierConfig

    def get(self, tier_name: str) -> TierConfig:
        """Return the TierConfig for a tier name string like 'tier_1'."""
        return getattr(self, tier_name)


class TelemetryConfig(BaseModel):
    """Logging / telemetry configuration."""

    log_file: str = "logs/routing_telemetry.jsonl"
    max_bytes: int = Field(10_485_760, gt=0, description="Max JSONL log file size before rotation (bytes).")
    backup_count: int = Field(5, ge=0, description="Number of rotated log files to retain.")
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------

class CascadeConfig(BaseModel):
    """Root configuration object for the entire cascade router service."""

    routing: RoutingConfig = RoutingConfig()
    gatekeeper_1: GatekeeperConfig
    gatekeeper_2: GatekeeperConfig
    tiers: TiersConfig
    telemetry: TelemetryConfig = TelemetryConfig()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path = "config/cascade_config.yaml") -> CascadeConfig:
    """
    Load, parse, and validate the YAML config file.

    Args:
        path: Path to cascade_config.yaml (default: config/cascade_config.yaml,
              relative to the working directory when the service is started).

    Returns:
        A fully validated CascadeConfig instance.

    Raises:
        FileNotFoundError: if the config file does not exist.
        pydantic.ValidationError: if required fields are missing or invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path.resolve()!s}\n"
            "Ensure you are running from the project root, or pass an explicit path."
        )
    raw: dict = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return CascadeConfig.model_validate(raw)
