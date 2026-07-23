"""
cascade.py
Two-gate escalation logic — the heart of the cascade router.

Architecture
------------

    Query ──► Gatekeeper 1 (G1)
               │  score < threshold_1  →  Tier 1 (STOP — G2 never called)
               └  score ≥ threshold_1  →  Gatekeeper 2 (G2)
                                            │  score < threshold_2  →  Tier 2 (STOP)
                                            └  score ≥ threshold_2  →  Tier 3 (STOP)

Key design choices
------------------
* Short-circuit: if G1 decides Tier 1, G2 is **never invoked** and never
  imports or touches the gatekeeper_2 Controller. This is verified via
  logging — look for "G2 not invoked" in the telemetry output.

* Explicit thresholds: the raw score from Controller.batch_calculate_win_rate()
  is compared against the configured threshold in plain Python (score >= threshold).
  RouteLLM's Controller.route() method is NOT used here because it would apply
  the threshold internally, hiding the score from our logs.

* Fail-safe: on any exception or timeout from a Controller call, the router
  escalates (goes UP a tier), never silently downgrades. The fail-safe event
  is logged at WARNING level and recorded in telemetry.

* Timeout: each Controller call runs in a ThreadPoolExecutor with
  config.routing.timeout_seconds. If the router hangs (e.g. network to a
  remote model server), the fail-safe fires.
"""
from __future__ import annotations

import concurrent.futures
import logging
import time
import uuid
from dataclasses import dataclass, field

from .config_loader import CascadeConfig
from .router_pool import RouterPool, get_score
from .telemetry import TelemetryLogger

logger = logging.getLogger(__name__)

# Tier progression used by the fail-safe escalation logic.
_TIER_ORDER = ["tier_1", "tier_2", "tier_3"]


# ---------------------------------------------------------------------------
# RoutingResult — per-query record
# ---------------------------------------------------------------------------

@dataclass
class RoutingResult:
    """
    Immutable record of a single routing decision.

    Every field here is written to the JSONL telemetry log by TelemetryLogger.
    Use these logs to analyse score distributions and tune thresholds offline.
    """

    query_id: str
    """UUID4 identifying this query for correlation with generation logs."""

    prompt: str
    """The original user query (stored for preview in telemetry)."""

    tier: str
    """Final tier chosen: 'tier_1', 'tier_2', or 'tier_3'."""

    tier_label: str
    """Human-readable label from config (e.g. 'Small', 'Medium', 'Large')."""

    gatekeepers_fired: list[str]
    """Which gatekeepers were invoked. G2 never appears when G1 → Tier 1."""

    g1_score: float | None
    """Win-probability score from Gatekeeper 1. None if G1 errored."""

    g2_score: float | None
    """Win-probability score from Gatekeeper 2. None if G2 wasn't invoked or errored."""

    routing_latency_ms: float
    """Wall-clock time spent in routing only (excludes LLM generation time)."""

    fail_safe_triggered: bool
    """True if any Controller call failed and the fail-safe escalation fired."""


# ---------------------------------------------------------------------------
# CascadeRouter
# ---------------------------------------------------------------------------

class CascadeRouter:
    """
    Two-stage cascade router.

    Instantiate once at service startup; call `route()` for each incoming query.

    Example
    -------
        from modelcascader import CascadeRouter, load_config, build_router_pool
        from modelcascader.telemetry import TelemetryLogger

        config = load_config("config/cascade_config.yaml")
        pool   = build_router_pool(config)
        tel    = TelemetryLogger(config.telemetry)
        router = CascadeRouter(config, pool, tel)

        result = router.route("Explain the Riemann hypothesis in simple terms.")
        print(result.tier)         # e.g. "tier_2"
        print(result.g1_score)     # e.g. 0.183
    """

    def __init__(
        self,
        config: CascadeConfig,
        pool: RouterPool,
        telemetry: TelemetryLogger,
    ) -> None:
        self.config = config
        self.pool = pool
        self.telemetry = telemetry
        self._timeout = config.routing.timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, prompt: str) -> RoutingResult:
        """
        Route a single query through the two-gate cascade.

        Args:
            prompt: The raw user query string.

        Returns:
            A RoutingResult with the chosen tier and all diagnostic fields.
            The result is also written to the telemetry log before returning.
        """
        query_id = str(uuid.uuid4())
        start = time.perf_counter()
        fail_safe = False
        gatekeepers_fired: list[str] = []
        g1_score: float | None = None
        g2_score: float | None = None

        # ── Gatekeeper 1 ─────────────────────────────────────────────────────
        gatekeepers_fired.append("gatekeeper_1")
        g1_score, g1_failed = self._safe_score(
            controller=self.pool.gatekeeper_1,
            router_name=self.config.gatekeeper_1.router,
            prompt=prompt,
            label="G1",
            query_id=query_id,
        )

        if g1_failed:
            # G1 errored — fail-safe: escalate past Tier 1 into G2
            fail_safe = True
            logger.warning(
                "[%s]  G1 failed — fail-safe: escalating past Tier 1, proceeding to G2.",
                query_id[:8],
            )
        elif g1_score < self.config.gatekeeper_1.threshold:  # type: ignore[operator]
            # ── Short-circuit: route to Tier 1, G2 is never called ───────────
            logger.debug(
                "[%s]  G1 score=%.4f < threshold=%.4f → Tier 1. G2 not invoked.",
                query_id[:8], g1_score, self.config.gatekeeper_1.threshold,
            )
            return self._finalise(
                query_id=query_id,
                prompt=prompt,
                tier="tier_1",
                gatekeepers_fired=gatekeepers_fired,
                g1_score=g1_score,
                g2_score=None,
                fail_safe=False,
                start=start,
            )

        # ── Gatekeeper 2 (only reached on G1 escalation or G1 failure) ───────
        if not g1_failed:
            logger.debug(
                "[%s]  G1 score=%.4f ≥ threshold=%.4f → escalating to G2.",
                query_id[:8], g1_score, self.config.gatekeeper_1.threshold,
            )

        gatekeepers_fired.append("gatekeeper_2")
        g2_score, g2_failed = self._safe_score(
            controller=self.pool.gatekeeper_2,
            router_name=self.config.gatekeeper_2.router,
            prompt=prompt,
            label="G2",
            query_id=query_id,
        )

        if g2_failed:
            # G2 errored — fail-safe: escalate to Tier 3
            fail_safe = True
            tier = "tier_3"
            logger.warning(
                "[%s]  G2 failed — fail-safe: escalating to Tier 3.",
                query_id[:8],
            )
        else:
            tier = (
                "tier_2"
                if g2_score < self.config.gatekeeper_2.threshold  # type: ignore[operator]
                else "tier_3"
            )
            logger.debug(
                "[%s]  G2 score=%.4f vs threshold=%.4f → %s.",
                query_id[:8],
                g2_score,
                self.config.gatekeeper_2.threshold,
                tier,
            )

        return self._finalise(
            query_id=query_id,
            prompt=prompt,
            tier=tier,
            gatekeepers_fired=gatekeepers_fired,
            g1_score=g1_score,
            g2_score=g2_score,
            fail_safe=fail_safe,
            start=start,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_score(
        self,
        controller,
        router_name: str,
        prompt: str,
        label: str,
        query_id: str,
    ) -> tuple[float | None, bool]:
        """
        Call get_score() in a thread with a timeout.

        Returns (score, failed) where failed=True means an exception or
        timeout occurred. The caller is responsible for triggering the
        fail-safe.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_score, controller, router_name, prompt)
            try:
                score = future.result(timeout=self._timeout)
                return score, False
            except concurrent.futures.TimeoutError:
                logger.error(
                    "[%s]  %s timed out after %.1f s.",
                    query_id[:8], label, self._timeout,
                )
                return None, True
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[%s]  %s raised an error: %s",
                    query_id[:8], label, exc,
                )
                return None, True

    def _finalise(
        self,
        query_id: str,
        prompt: str,
        tier: str,
        gatekeepers_fired: list[str],
        g1_score: float | None,
        g2_score: float | None,
        fail_safe: bool,
        start: float,
    ) -> RoutingResult:
        latency_ms = (time.perf_counter() - start) * 1000
        tier_cfg = self.config.tiers.get(tier)
        result = RoutingResult(
            query_id=query_id,
            prompt=prompt,
            tier=tier,
            tier_label=tier_cfg.label,
            gatekeepers_fired=gatekeepers_fired,
            g1_score=g1_score,
            g2_score=g2_score,
            routing_latency_ms=latency_ms,
            fail_safe_triggered=fail_safe,
        )
        self.telemetry.log(result)
        return result
