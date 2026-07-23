"""
server.py
Local test/debug server for the Three-Tier Cascade Router.

Serves two things:
  POST /route-and-generate  — runs the cascade router then calls the winning
                               tier's LLM and returns a single JSON response.
  GET  /                    — serves frontend/index.html as the UI.
  GET  /frontend/*          — serves static files from the frontend/ directory.

Usage
-----
    # From the project root, with the venv active:
    python server.py

    # With an OpenAI key (required for any tier using provider: openai):
    set OPENAI_API_KEY=sk-...       # Windows
    export OPENAI_API_KEY=sk-...    # macOS/Linux
    python server.py

    # Optional: change the port (default 8765)
    python server.py --port 9000

Then open http://localhost:8765 in your browser.

Architecture
------------
Uses only Python stdlib (http.server + json + threading) — no Flask/FastAPI
needed. This is intentional: the test UI is a debug tool, not a service.

The routing path calls the existing CascadeRouter.route() and providers.generate()
without duplicating any logic. The server is single-threaded (one request at a
time), which is fine for a single-user debug tool.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── Bootstrap: ensure project root is on sys.path ────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Set placeholder OPENAI_API_KEY before RouteLLM import ────────────────────
# RouteLLM's similarity_weighted router instantiates OpenAI() at module import
# time. Setting a placeholder satisfies the constructor without making real calls.
# If a real key is already set, this no-op guard leaves it untouched.
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "placeholder-for-routellm-import"

from modelcascader import CascadeRouter, load_config, build_router_pool
from modelcascader.telemetry import TelemetryLogger
from modelcascader.providers import get_client, generate

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

# ── Global singletons (initialised once at startup) ───────────────────────────
_config = None
_router = None
_frontend_dir = ROOT / "frontend"


def _init_router():
    global _config, _router
    logger.info("Loading config...")
    _config = load_config(ROOT / "config" / "cascade_config.yaml")
    pool = build_router_pool(_config)
    tel = TelemetryLogger(_config.telemetry)
    _router = CascadeRouter(_config, pool, tel)
    logger.info("Cascade router ready.")


# ── Request handler ───────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — routes two paths only."""

    def log_message(self, format, *args):  # suppress default access log noise
        logger.debug("HTTP %s %s", self.path, args)

    # ── GET — serve static files ──────────────────────────────────────────────

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_file(_frontend_dir / "index.html", "text/html; charset=utf-8")
        elif self.path.startswith("/frontend/"):
            rel = self.path.lstrip("/")
            self._serve_file(ROOT / rel, "application/octet-stream")
        else:
            self._send_json(404, {"error": f"Not found: {self.path}"})

    def _serve_file(self, path: Path, content_type: str):
        if not path.exists():
            self._send_json(404, {"error": f"File not found: {path}"})
            return
        data = path.read_bytes()
        # Guess content type from extension
        if path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ── POST /route-and-generate ──────────────────────────────────────────────

    def do_POST(self):
        if self.path != "/route-and-generate":
            self._send_json(404, {"error": f"Unknown endpoint: {self.path}"})
            return

        # Parse request body
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"Invalid JSON: {exc}"})
            return

        prompt = payload.get("prompt", "").strip()
        if not prompt:
            self._send_json(400, {"error": "Missing or empty 'prompt' field."})
            return

        try:
            response_data = _handle_route_and_generate(prompt)
            self._send_json(200, response_data)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.error("Unhandled error in /route-and-generate:\n%s", tb)
            self._send_json(500, {"error": str(exc), "traceback": tb})

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # Allow the frontend to call this from file:// origin in a browser
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Pre-flight CORS for browsers that send OPTIONS first."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ── Core handler logic ────────────────────────────────────────────────────────

def _handle_route_and_generate(prompt: str) -> dict:
    """
    Run the cascade router, then call the winning tier's LLM.

    Returns a dict matching the API contract documented in the module docstring.
    Raises on unrecoverable errors (caller converts to HTTP 500).
    """
    # ── Step 1: routing ───────────────────────────────────────────────────────
    logger.info("Routing prompt (%.60s...)", prompt)
    routing_result = _router.route(prompt)

    tier = routing_result.tier                         # e.g. "tier_1"
    tier_cfg = _config.tiers.get(tier)                 # TierConfig
    g1_threshold = _config.gatekeeper_1.threshold
    g2_threshold = _config.gatekeeper_2.threshold

    logger.info(
        "Routed to %s (%s) via %s  g1=%.4f  g2=%s",
        tier, tier_cfg.model,
        routing_result.gatekeepers_fired,
        routing_result.g1_score or 0.0,
        f"{routing_result.g2_score:.4f}" if routing_result.g2_score is not None else "N/A",
    )

    # ── Step 2: generation ────────────────────────────────────────────────────
    gen_start = time.perf_counter()
    gen_error: str | None = None
    response_text: str | None = None

    try:
        client = get_client(tier_cfg)
        messages = [{"role": "user", "content": prompt}]
        response_text = generate(client, tier_cfg, messages)
        logger.info("Generation complete (%.1f ms)", (time.perf_counter() - gen_start) * 1000)
    except Exception as exc:  # noqa: BLE001
        gen_error = _friendly_error(exc, tier_cfg)
        logger.error("Generation failed: %s", gen_error)

    generation_latency_ms = (time.perf_counter() - gen_start) * 1000

    # ── Step 3: assemble response ─────────────────────────────────────────────
    resp: dict = {
        "final_tier": tier,
        "tier_label": tier_cfg.label,
        "model_used": tier_cfg.model,
        "provider": tier_cfg.provider,
        "g1_score": routing_result.g1_score,
        "g2_score": routing_result.g2_score,
        "g1_threshold": g1_threshold,
        "g2_threshold": g2_threshold,
        "gatekeepers_fired": routing_result.gatekeepers_fired,
        "fail_safe_triggered": routing_result.fail_safe_triggered,
        "routing_latency_ms": round(routing_result.routing_latency_ms, 1),
        "generation_latency_ms": round(generation_latency_ms, 1),
        "response_text": response_text,
        "error": gen_error,
    }
    return resp


def _friendly_error(exc: Exception, tier_cfg) -> str:
    """Turn provider exceptions into human-readable messages that name the missing key."""
    msg = str(exc)
    provider = tier_cfg.provider

    KEY_NAMES = {
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq":      "GROQ_API_KEY",
        "google":    "GEMINI_API_KEY (note: GOOGLE_API_KEY takes precedence if also set)",
    }

    # Auth / missing key errors
    auth_signals = ("api_key", "authentication", "401", "x-api-key",
                    "invalid api key", "api_key_missing", "unauthenticated")
    if any(s in msg.lower() for s in auth_signals):
        key_hint = KEY_NAMES.get(provider, f"the API key for '{provider}'")
        return (
            f"Authentication failed for provider '{provider}' "
            f"(model: {tier_cfg.model}). "
            f"Set {key_hint} in your environment."
        )
    return f"Generation failed ({provider}/{tier_cfg.model}): {msg}"


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cascade Router local test server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0 — listens on all interfaces). "
             "Use 127.0.0.1 to restrict to localhost only.",
    )
    args = parser.parse_args()

    _init_router()

    server = HTTPServer((args.host, args.port), Handler)
    logger.info("Server ready at http://%s:%d", args.host, args.port)
    logger.info("Open the above URL in your browser to use the test UI.")
    logger.info("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
