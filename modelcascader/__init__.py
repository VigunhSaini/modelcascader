"""
modelcascader — Three-Tier Model Cascade Router

Public API surface:
    CascadeRouter   — main orchestration object (cascade.py)
    RoutingResult   — per-query routing record (cascade.py)
    load_config     — load & validate cascade_config.yaml (config_loader.py)
    build_router_pool — initialise RouteLLM Controllers (router_pool.py)
    TelemetryLogger — structured JSONL logger (telemetry.py)
"""

from .cascade import CascadeRouter, RoutingResult
from .config_loader import CascadeConfig, load_config
from .router_pool import RouterPool, build_router_pool
from .telemetry import TelemetryLogger

__all__ = [
    "CascadeRouter",
    "RoutingResult",
    "CascadeConfig",
    "load_config",
    "RouterPool",
    "build_router_pool",
    "TelemetryLogger",
]
