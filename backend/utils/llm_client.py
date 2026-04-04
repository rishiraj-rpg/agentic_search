"""
Thin async wrapper around the OpenAI client.
Provides two helpers:
  - chat_text  → plain string response
  - chat_json  → parsed JSON (uses JSON mode to avoid markdown fences)
"""
from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI

from utils.cost_tracker import record_usage

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


async def chat_text(system: str, user: str, model: str = "gpt-4o-mini") -> str:
    """Return a plain text LLM response."""
    resp = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.2,
    )
    if resp.usage:
        record_usage(
            model=model,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            total_tokens=resp.usage.total_tokens,
        )
    return resp.choices[0].message.content or ""


async def chat_json(system: str, user: str, model: str = "gpt-4o-mini") -> Any:
    """
    Return a parsed JSON object from the LLM.
    JSON mode is enforced so the model never wraps output in markdown fences.
    """
    resp = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    if resp.usage:
        record_usage(
            model=model,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            total_tokens=resp.usage.total_tokens,
        )
    raw = resp.choices[0].message.content or "{}"
    return json.loads(raw)
