from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.config import get_settings


@dataclass
class LLMResult:
    text: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """OpenAI-compatible chat client. Generator and judge share this; models are chosen per call."""

    def __init__(self, role: str = "generator") -> None:
        s = get_settings()
        self.base_url = s.llm_base_url.rstrip("/")
        self.api_key = s.llm_api_key
        self.role = role
        self.default_model = s.generator_model if role == "generator" else s.judge_model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> LLMResult:
        model = model or self.default_model
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        t0 = time.perf_counter()
        with httpx.Client(timeout=180.0) as client:
            r = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        latency_ms = (time.perf_counter() - t0) * 1000
        choice = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or _estimate_tokens(messages))
        completion_tokens = int(usage.get("completion_tokens") or _estimate_tokens(choice))
        return LLMResult(
            text=choice,
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw=data,
        )


def _estimate_tokens(obj: Any) -> int:
    if isinstance(obj, list):
        text = json.dumps(obj)
    else:
        text = str(obj)
    return max(1, len(text) // 4)
