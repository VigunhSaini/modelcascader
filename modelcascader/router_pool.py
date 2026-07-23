"""
router_pool.py
Initialise RouteLLM Controller instances and provide a clean,
Controller-level score-extraction interface.

Design notes
------------
* Two Controller instances are created at startup — one per gatekeeper —
  and reused across all requests. Controller init is not cheap (it loads
  pretrained router weights) so we do it exactly once.

* Score extraction calls Controller.batch_calculate_win_rate() with a
  single-item list for live per-query routing. This is consistent with
  the batch path used by the eval harness. No `.router` sub-attribute
  is accessed anywhere — all calls go through the public Controller API.

* The `get_score` function is intentionally a module-level function, not
  a method, so it can be passed to a ThreadPoolExecutor for timeout
  enforcement in cascade.py without binding `self`.

* OPENAI_API_KEY guard: RouteLLM's similarity_weighted router calls
  OpenAI() at module import time (a library bug). We set a placeholder
  key before importing so the constructor doesn't raise — the mf router
  never actually contacts OpenAI during scoring. If you have a real key
  set in the environment already, this no-op guard leaves it untouched.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import pandas as pd

# RouteLLM's similarity_weighted/utils.py instantiates OpenAI() at module
# level, which raises immediately if OPENAI_API_KEY is unset — even when
# the mf router is used and no OpenAI calls are ever made during routing.
# Setting a placeholder here satisfies the constructor without side effects.
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "placeholder-for-routellm-import"

from routellm.controller import Controller

from .config_loader import CascadeConfig, GatekeeperConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RouterPool dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouterPool:
    """Holds both initialised Controller instances. Immutable after creation."""

    gatekeeper_1: Controller
    gatekeeper_2: Controller


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_router_pool(config: CascadeConfig) -> RouterPool:
    """
    Instantiate both RouteLLM Controllers from config.

    Call this once at service startup. Each Controller loads pretrained
    router weights on first use (mf: ~50 MB, bert: larger).

    Args:
        config: Validated CascadeConfig.

    Returns:
        A frozen RouterPool holding gatekeeper_1 and gatekeeper_2.
    """
    logger.info("Building router pool…")
    g1 = _make_controller(config.gatekeeper_1, label="gatekeeper_1")
    g2 = _make_controller(config.gatekeeper_2, label="gatekeeper_2")
    logger.info("Router pool ready.")
    return RouterPool(gatekeeper_1=g1, gatekeeper_2=g2)


def _make_controller(cfg: GatekeeperConfig, label: str) -> Controller:
    logger.info(
        "Initialising %s  [router=%s  weak=%s  strong=%s]",
        label, cfg.router, cfg.weak_model, cfg.strong_model,
    )
    return Controller(
        routers=[cfg.router],
        weak_model=cfg.weak_model,
        strong_model=cfg.strong_model,
    )


# ---------------------------------------------------------------------------
# Score extraction — public interface
# ---------------------------------------------------------------------------

def get_score(controller: Controller, router_name: str, prompt: str) -> float:
    """
    Return the win-probability score for a single prompt.

    Wraps the prompt in a pandas Series before calling
    Controller.batch_calculate_win_rate(), which internally calls
    Series.apply() and therefore requires a Series, not a plain list.

    Args:
        controller:  The RouteLLM Controller for the relevant gatekeeper.
        router_name: Name of the router to use (e.g. "mf", "bert").
        prompt:      The raw user query string.

    Returns:
        Float in [0, 1].

    Raises:
        Any exception from Controller.batch_calculate_win_rate() — callers
        (cascade.py) are responsible for catching and triggering the fail-safe.
    """
    scores = controller.batch_calculate_win_rate(
        prompts=pd.Series([prompt]),
        router=router_name,
    )
    # Returns a pandas Series; extract the single scalar value.
    return float(scores.iloc[0])


def batch_get_scores(
    controller: Controller,
    router_name: str,
    prompts: list[str],
) -> list[float]:
    """
    Return win-probability scores for a list of prompts.

    Converts the list to a pandas Series before calling
    Controller.batch_calculate_win_rate(), which requires a Series
    (it calls .apply() internally).

    Args:
        controller:  The RouteLLM Controller to query.
        router_name: Name of the router to use.
        prompts:     List of raw query strings.

    Returns:
        List of floats in [0, 1], one per prompt, in the same order.
    """
    scores = controller.batch_calculate_win_rate(
        prompts=pd.Series(prompts),
        router=router_name,
    )
    return [float(s) for s in scores]
