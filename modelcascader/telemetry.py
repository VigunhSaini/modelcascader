"""
telemetry.py
Structured per-query telemetry: JSONL rotating log + human-readable console output.

Each routing event is recorded as a single JSON object (one per line) containing
every field needed for later threshold analysis and cost accounting.

Log schema
----------
{
    "ts":                  "2024-01-15T10:23:45.123456Z",   # ISO-8601 UTC
    "query_id":            "a3f1...",                        # UUID4
    "prompt_preview":      "What is the capital…",           # first 80 chars
    "gatekeepers_fired":   ["gatekeeper_1"],                 # which gates ran
    "g1_score":            0.083,                            # float | null
    "g2_score":            null,                             # float | null
    "final_tier":          "tier_1",                         # "tier_1"|"tier_2"|"tier_3"
    "tier_label":          "Small",
    "routing_latency_ms":  4.2,
    "fail_safe_triggered": false
}
"""
from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cascade import RoutingResult


# ---------------------------------------------------------------------------
# TelemetryLogger
# ---------------------------------------------------------------------------

class TelemetryLogger:
    """
    Writes per-query routing records to a rotating JSONL file and
    mirrors human-readable summaries to the console logger.

    Usage
    -----
        from modelcascader.telemetry import TelemetryLogger
        from modelcascader.config_loader import TelemetryConfig

        tel = TelemetryLogger(config.telemetry)
        tel.log(routing_result)
    """

    def __init__(self, config: "TelemetryConfig") -> None:  # noqa: F821
        # ── Console logger ────────────────────────────────────────────────
        self._console = logging.getLogger("modelcascader.routing")
        if not self._console.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%H:%M:%S",
            ))
            self._console.addHandler(handler)
        self._console.setLevel(getattr(logging, config.console_level, logging.INFO))

        # ── Rotating JSONL log ────────────────────────────────────────────
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self._jsonl_logger = logging.getLogger("modelcascader.telemetry.jsonl")
        self._jsonl_logger.propagate = False  # don't echo to console
        if not self._jsonl_logger.handlers:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_path,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding="utf-8",
            )
            # Bare formatter — the message IS the JSON object
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            self._jsonl_logger.addHandler(file_handler)
        self._jsonl_logger.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    def log(self, result: "RoutingResult") -> None:
        """Record a completed routing decision to JSONL + console."""
        record = self._build_record(result)
        self._jsonl_logger.debug(json.dumps(record, separators=(",", ":")))
        self._log_console(result, record)

    # ------------------------------------------------------------------
    def _build_record(self, r: "RoutingResult") -> dict:
        return {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "query_id": r.query_id,
            "prompt_preview": r.prompt[:80] + ("…" if len(r.prompt) > 80 else ""),
            "gatekeepers_fired": r.gatekeepers_fired,
            "g1_score": round(r.g1_score, 6) if r.g1_score is not None else None,
            "g2_score": round(r.g2_score, 6) if r.g2_score is not None else None,
            "final_tier": r.tier,
            "tier_label": r.tier_label,
            "routing_latency_ms": round(r.routing_latency_ms, 3),
            "fail_safe_triggered": r.fail_safe_triggered,
        }

    def _log_console(self, r: "RoutingResult", record: dict) -> None:
        fail_tag = "  [FAIL-SAFE]" if r.fail_safe_triggered else ""
        score_parts = []
        if r.g1_score is not None:
            score_parts.append(f"G1={r.g1_score:.3f}")
        if r.g2_score is not None:
            score_parts.append(f"G2={r.g2_score:.3f}")
        scores_str = "  ".join(score_parts) or "no scores"

        self._console.info(
            "[%s]  → %s (%s)  |  %s  |  %.1f ms%s",
            r.query_id[:8],
            r.tier,
            r.tier_label,
            scores_str,
            r.routing_latency_ms,
            fail_tag,
        )
        if r.fail_safe_triggered:
            self._console.warning(
                "[%s]  Fail-safe fired — routing escalated to %s. "
                "Check logs for the upstream error.",
                r.query_id[:8],
                r.tier,
            )
