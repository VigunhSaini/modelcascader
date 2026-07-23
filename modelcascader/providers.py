"""
providers.py
Thin dispatch layer: given a TierConfig, return the right provider client
and a generate() function.

This module is the ONLY place that knows about provider SDKs.
Routing logic (cascade.py) never imports openai or anthropic directly,
so swapping a provider means editing config/cascade_config.yaml only.

Supported providers
-------------------
  openai     — OpenAI Chat Completions API (OPENAI_API_KEY)
  anthropic  — Anthropic Messages API (ANTHROPIC_API_KEY)
  groq       — Groq API, OpenAI-compatible (GROQ_API_KEY)
  google     — Google Gemini API via google-genai SDK (GEMINI_API_KEY)

               ⚠ google-genai env var precedence:
               The google-genai SDK checks GOOGLE_API_KEY first, then
               GEMINI_API_KEY. If a stray GOOGLE_API_KEY is set in your
               environment (e.g. left over from another tool), it will be
               used silently instead of GEMINI_API_KEY, potentially causing
               confusing authentication failures. Unset GOOGLE_API_KEY or
               set it to the same value as GEMINI_API_KEY to avoid this.

Usage
-----
    from modelcascader.providers import get_client, generate

    client = get_client(config.tiers.tier_2)
    response_text = generate(client, config.tiers.tier_2, messages)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config_loader import TierConfig

logger = logging.getLogger(__name__)

_ClientType = Any


def get_client(tier: "TierConfig") -> _ClientType:
    """
    Return an initialised provider client for the given tier.

    Args:
        tier: TierConfig from the validated YAML (provider + model).

    Returns:
        A provider client object.

    Raises:
        ValueError: if the provider is not supported.
        ImportError: if the provider's SDK is not installed.
    """
    if tier.provider == "openai":
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai package not installed. Run: pip install openai") from exc
        client = openai.OpenAI()
        logger.debug("OpenAI client ready for model=%s", tier.model)
        return client

    if tier.provider == "anthropic":
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("anthropic package not installed. Run: pip install anthropic") from exc
        client = anthropic.Anthropic()
        logger.debug("Anthropic client ready for model=%s", tier.model)
        return client

    if tier.provider == "groq":
        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError("groq package not installed. Run: pip install groq") from exc
        client = Groq()   # reads GROQ_API_KEY from environment
        logger.debug("Groq client ready for model=%s", tier.model)
        return client

    if tier.provider == "google":
        try:
            from google import genai as google_genai
        except ImportError as exc:
            raise ImportError(
                "google-genai package not installed. Run: pip install google-genai"
            ) from exc
        # google-genai checks GOOGLE_API_KEY first, then GEMINI_API_KEY.
        # We construct the client explicitly so the precedence is visible.
        import os
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Google API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) "
                "in your environment. Note: GOOGLE_API_KEY takes precedence over "
                "GEMINI_API_KEY if both are set."
            )
        client = google_genai.Client(api_key=api_key)
        logger.debug("Google Gemini client ready for model=%s", tier.model)
        return client

    raise ValueError(
        f"Unsupported provider: '{tier.provider}'. "
        "Valid options: 'openai', 'anthropic', 'groq', 'google'. "
        "To add a provider, extend modelcascader/providers.py and add it to config_loader.py."
    )


def generate(
    client: _ClientType,
    tier: "TierConfig",
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> str:
    """
    Call the provider API and return the assistant's response text.

    Args:
        client:   Provider client returned by get_client().
        tier:     TierConfig (used to look up the model identifier).
        messages: OpenAI-format message list, e.g.:
                  [{"role": "user", "content": "Hello"}]
        **kwargs: Extra provider kwargs forwarded to the completion call.

    Returns:
        The response content as a plain string.

    Raises:
        RuntimeError: on API error, with the original exception chained.
    """
    provider = tier.provider

    if provider == "openai":
        return _generate_openai(client, tier.model, messages, **kwargs)

    if provider == "anthropic":
        return _generate_anthropic(client, tier.model, messages, **kwargs)

    if provider == "groq":
        return _generate_groq(client, tier.model, messages, **kwargs)

    if provider == "google":
        return _generate_google(client, tier.model, messages, **kwargs)

    raise ValueError(f"Unsupported provider: '{provider}'")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _generate_openai(client: Any, model: str, messages: list[dict], **kwargs) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.error("OpenAI API error [model=%s]: %s", model, exc)
        raise RuntimeError(f"OpenAI generation failed: {exc}") from exc


def _generate_anthropic(client: Any, model: str, messages: list[dict], **kwargs) -> str:
    """
    Anthropic uses a slightly different message schema:
    system messages must be extracted and passed separately.
    """
    try:
        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        user_messages = [m for m in messages if m["role"] != "system"]
        system_text = "\n".join(system_messages) if system_messages else None

        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "messages": user_messages,
            **kwargs,
        }
        if system_text:
            create_kwargs["system"] = system_text

        response = client.messages.create(**create_kwargs)
        return response.content[0].text if response.content else ""
    except Exception as exc:
        logger.error("Anthropic API error [model=%s]: %s", model, exc)
        raise RuntimeError(f"Anthropic generation failed: {exc}") from exc


def _generate_groq(client: Any, model: str, messages: list[dict], **kwargs) -> str:
    """
    Groq's API is OpenAI-compatible, so the call shape is identical.
    The Groq SDK wraps the same Chat Completions interface.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.error("Groq API error [model=%s]: %s", model, exc)
        raise RuntimeError(f"Groq generation failed: {exc}") from exc


def _generate_google(client: Any, model: str, messages: list[dict], **kwargs) -> str:
    """
    Google Gemini via google-genai SDK.

    google-genai uses a different call shape from OpenAI:
      - System messages are extracted and passed as system_instruction.
      - The remaining messages are converted to google-genai Content objects.
      - Response text is accessed via response.text.
    """
    try:
        from google.genai import types as genai_types

        # Extract system instruction (Gemini passes it separately)
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        system_instruction = "\n".join(system_parts) if system_parts else None

        # Convert remaining messages to Gemini Content objects
        # Gemini roles: "user" and "model" (not "assistant")
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part(text=m["content"])],
                )
            )

        config_kwargs: dict[str, Any] = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
        )
        return response.text or ""
    except Exception as exc:
        logger.error("Google Gemini API error [model=%s]: %s", model, exc)
        raise RuntimeError(f"Google Gemini generation failed: {exc}") from exc
