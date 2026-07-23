"""
evaluate.py
Evaluation harness — tier distribution report over a sample query set.

This script runs routing decisions only: it calls the RouteLLM Controllers
to get scores and applies thresholds, but makes NO LLM generation calls
and therefore requires NO provider API keys.

Usage
-----
    python eval/evaluate.py                              # uses defaults
    python eval/evaluate.py --queries sample_queries.txt --config config/cascade_config.yaml
    python eval/evaluate.py --queries my_queries.txt --out results/eval_run.jsonl

Output (to terminal)
--------------------
    ┌────────────────────────────────────────────────────────┐
    │     Cascade Router — Tier Distribution Report          │
    ├────────────┬──────────┬────────────┬───────────────────┤
    │ Tier       │  Count   │     %      │  Mean G-score     │
    ├────────────┼──────────┼────────────┼───────────────────┤
    │ Tier 1     │    42    │  56.0 %    │  G1: 0.047        │
    │ Tier 2     │    25    │  33.3 %    │  G1: 0.204  G2: 0.153 │
    │ Tier 3     │     8    │  10.7 %    │  G1: 0.231  G2: 0.491 │
    ├────────────┼──────────┼────────────┼───────────────────┤
    │ TOTAL      │    75    │ 100.0 %    │                   │
    └────────────┴──────────┴────────────┴───────────────────┘
    Fail-safes: 0 / 75

Strategy
--------
The eval harness uses Controller.batch_calculate_win_rate() for each
gatekeeper (RouteLLM's designated batch path), which is more efficient
than calling get_score() in a loop for large query sets.

Specifically:
  1. Run ALL queries through G1 batch → split by threshold_1
  2. Run ESCALATED queries through G2 batch → split by threshold_2
This means G2 is called only once (as a batch), not once per query.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modelcascader.config_loader import load_config
from modelcascader.router_pool import batch_get_scores, build_router_pool
from modelcascader.telemetry import TelemetryLogger

try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    queries: list[str],
    config_path: str = "config/cascade_config.yaml",
    output_path: str | None = None,
) -> list[dict]:
    """
    Run routing decisions for all queries and return per-query result dicts.

    No LLM generation is performed; only the two RouteLLM Controllers are
    called (via batch_calculate_win_rate) to obtain scores.

    Args:
        queries:     List of raw query strings to evaluate.
        config_path: Path to cascade_config.yaml.
        output_path: If provided, write per-query JSONL results here.

    Returns:
        List of result dicts (one per query), suitable for further analysis.
    """
    config = load_config(config_path)
    pool = build_router_pool(config)

    t1_threshold = config.gatekeeper_1.threshold
    t2_threshold = config.gatekeeper_2.threshold
    router_1 = config.gatekeeper_1.router
    router_2 = config.gatekeeper_2.router

    n = len(queries)
    print(f"\nEvaluating {n} queries against cascade thresholds...")
    print(f"  G1 threshold = {t1_threshold}  (router: {router_1})")
    print(f"  G2 threshold = {t2_threshold}  (router: {router_2})\n")

    # -- Step 1: score ALL queries through Gatekeeper 1 --
    print("-> Running Gatekeeper 1 batch...")
    g1_scores = batch_get_scores(pool.gatekeeper_1, router_1, queries)

    # Split: below threshold → Tier 1; else → escalate to G2
    tier1_indices = [i for i, s in enumerate(g1_scores) if s < t1_threshold]
    escalated_indices = [i for i, s in enumerate(g1_scores) if s >= t1_threshold]

    print(f"   {len(tier1_indices)} queries resolve at Tier 1 (score < {t1_threshold})")
    print(f"   {len(escalated_indices)} queries escalate to Gatekeeper 2")

    # ── Step 2: score ESCALATED queries through Gatekeeper 2 ─────────────────
    g2_scores_map: dict[int, float] = {}
    if escalated_indices:
        print("-> Running Gatekeeper 2 batch (escalated queries only)...")
        escalated_prompts = [queries[i] for i in escalated_indices]
        g2_scores_list = batch_get_scores(pool.gatekeeper_2, router_2, escalated_prompts)
        g2_scores_map = {idx: score for idx, score in zip(escalated_indices, g2_scores_list)}

    # ── Step 3: assemble per-query results ───────────────────────────────────
    results: list[dict] = []
    for i, prompt in enumerate(queries):
        g1_score = g1_scores[i]
        if i in g2_scores_map:
            g2_score = g2_scores_map[i]
            tier = "tier_3" if g2_score >= t2_threshold else "tier_2"
            gatekeepers_fired = ["gatekeeper_1", "gatekeeper_2"]
        else:
            g2_score = None
            tier = "tier_1"
            gatekeepers_fired = ["gatekeeper_1"]

        tier_cfg = config.tiers.get(tier)
        results.append({
            "index": i,
            "prompt": prompt,
            "prompt_preview": prompt[:80] + ("…" if len(prompt) > 80 else ""),
            "tier": tier,
            "tier_label": tier_cfg.label,
            "gatekeepers_fired": gatekeepers_fired,
            "g1_score": round(g1_score, 6),
            "g2_score": round(g2_score, 6) if g2_score is not None else None,
            "fail_safe_triggered": False,  # eval harness doesn't exercise fail-safe
        })

    # ── Step 4: write JSONL output ────────────────────────────────────────────
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nResults written to {out.resolve()}")

    return results


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(results: list[dict], config_path: str = "config/cascade_config.yaml") -> None:
    """Print a human-readable tier distribution table to the terminal."""
    config = load_config(config_path)
    n = len(results)
    if n == 0:
        print("No results to report.")
        return

    # Aggregate by tier
    tier_counts: dict[str, int] = defaultdict(int)
    tier_g1_scores: dict[str, list[float]] = defaultdict(list)
    tier_g2_scores: dict[str, list[float]] = defaultdict(list)
    fail_safe_count = 0

    for r in results:
        tier = r["tier"]
        tier_counts[tier] += 1
        tier_g1_scores[tier].append(r["g1_score"])
        if r["g2_score"] is not None:
            tier_g2_scores[tier].append(r["g2_score"])
        if r.get("fail_safe_triggered"):
            fail_safe_count += 1

    tier_order = ["tier_1", "tier_2", "tier_3"]
    tier_labels = {
        "tier_1": config.tiers.tier_1.label,
        "tier_2": config.tiers.tier_2.label,
        "tier_3": config.tiers.tier_3.label,
    }

    def mean(lst: list[float]) -> str:
        return f"{sum(lst) / len(lst):.4f}" if lst else "—"

    if _RICH:
        _print_rich_table(
            n, tier_order, tier_labels, tier_counts,
            tier_g1_scores, tier_g2_scores, fail_safe_count,
        )
    else:
        _print_plain_table(
            n, tier_order, tier_labels, tier_counts,
            tier_g1_scores, tier_g2_scores, fail_safe_count, mean,
        )


def _print_rich_table(n, tier_order, tier_labels, tier_counts,
                      tier_g1_scores, tier_g2_scores, fail_safe_count):
    console = Console()
    table = Table(
        title="Cascade Router — Tier Distribution Report",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Tier", style="bold")
    table.add_column("Model Label", style="dim")
    table.add_column("Count", justify="right")
    table.add_column("%", justify="right")
    table.add_column("Mean G1 score", justify="right")
    table.add_column("Mean G2 score", justify="right")

    def mean_str(lst):
        return f"{sum(lst) / len(lst):.4f}" if lst else "—"

    for tier in tier_order:
        count = tier_counts.get(tier, 0)
        pct = 100 * count / n if n else 0
        g1_mean = mean_str(tier_g1_scores.get(tier, []))
        g2_mean = mean_str(tier_g2_scores.get(tier, []))
        style = {"tier_1": "green", "tier_2": "yellow", "tier_3": "red"}.get(tier, "")
        table.add_row(
            tier, tier_labels.get(tier, "?"),
            str(count), f"{pct:.1f} %",
            g1_mean, g2_mean,
            style=style,
        )

    table.add_section()
    table.add_row("TOTAL", "", str(n), "100.0 %", "", "", style="bold")

    console.print()
    console.print(table)
    console.print(
        f"Fail-safes triggered: [bold red]{fail_safe_count}[/] / {n}\n"
        "(Fail-safes are 0 in the eval harness; they only appear during live routing.)"
    )
    console.print()


def _print_plain_table(n, tier_order, tier_labels, tier_counts,
                        tier_g1_scores, tier_g2_scores, fail_safe_count, mean):
    col_w = [12, 14, 8, 10, 16, 16]
    header = ["Tier", "Label", "Count", "%", "Mean G1", "Mean G2"]
    sep = "  ".join("-" * w for w in col_w)
    print("\n" + "Cascade Router — Tier Distribution Report".center(sum(col_w) + len(col_w) * 2))
    print(sep)
    print("  ".join(h.ljust(w) for h, w in zip(header, col_w)))
    print(sep)
    for tier in tier_order:
        count = tier_counts.get(tier, 0)
        pct = 100 * count / n if n else 0
        row = [
            tier,
            tier_labels.get(tier, "?"),
            str(count),
            f"{pct:.1f} %",
            mean(tier_g1_scores.get(tier, [])),
            mean(tier_g2_scores.get(tier, [])),
        ]
        print("  ".join(v.ljust(w) for v, w in zip(row, col_w)))
    print(sep)
    total_row = ["TOTAL", "", str(n), "100.0 %", "", ""]
    print("  ".join(v.ljust(w) for v, w in zip(total_row, col_w)))
    print(f"\nFail-safes: {fail_safe_count} / {n}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cascade Router Evaluation Harness\n"
            "Runs routing decisions (no LLM generation) over a query file "
            "and reports the tier distribution."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--queries", "-q",
        default="sample_queries.txt",
        help="Path to a plain-text file with one query per line (default: sample_queries.txt).",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/cascade_config.yaml",
        help="Path to cascade_config.yaml (default: config/cascade_config.yaml).",
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Optional path to write per-query JSONL results for offline analysis.",
    )
    return parser.parse_args()


def _load_queries(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: Query file not found: {p.resolve()}", file=sys.stderr)
        sys.exit(1)
    lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]
    queries = [q for q in lines if q and not q.startswith("#")]
    if not queries:
        print("ERROR: Query file is empty or contains only comments.", file=sys.stderr)
        sys.exit(1)
    return queries


if __name__ == "__main__":
    args = _parse_args()
    queries = _load_queries(args.queries)
    results = evaluate(queries, config_path=args.config, output_path=args.out)
    print_report(results, config_path=args.config)
