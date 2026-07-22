from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


EXPENSIVE_THINK = os.environ.get("EXPENSIVE_THINK", "0") == "1"
EXPENSIVE_NUM_PREDICT = int(
    os.environ.get("EXPENSIVE_NUM_PREDICT", "512" if EXPENSIVE_THINK else "128")
)


@dataclass
class ChatResult:
    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class ChatClient:
    """Small OpenAI-compatible/Ollama chat client.

    The official implementation uses the OpenAI SDK. This wrapper keeps the
    same chat-completion shape while allowing the local Ollama endpoint used by
    the SUQL project.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:11434",
        api_key: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        if model.startswith("ollama/") or self._is_ollama:
            return self._chat_ollama(
                messages, model, max_tokens, temperature, stop, response_schema
            )
        return self._chat_openai_compatible(
            messages, model, max_tokens, temperature, stop, response_schema
        )

    @property
    def _is_ollama(self) -> bool:
        return "11434" in self.api_base or self.api_base.endswith("/ollama")

    def _chat_ollama(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
        response_schema: dict[str, Any] | None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model.removeprefix("ollama/"),
            "messages": messages,
            "stream": False,
            "think": False if response_schema is not None else EXPENSIVE_THINK,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 4096,
            },
        }
        if stop:
            payload["options"]["stop"] = stop
        if response_schema is not None:
            payload["format"] = response_schema
        response = httpx.post(f"{self.api_base}/api/chat", json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        content = str(data.get("message", {}).get("content", ""))
        done_reason = str(data.get("done_reason", "stop"))
        finish_reason = "stop" if done_reason in {"stop", "unload"} else done_reason
        return ChatResult(
            content=content,
            finish_reason=finish_reason,
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
        )

    def _chat_openai_compatible(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
        response_schema: dict[str, Any] | None,
    ) -> ChatResult:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "semantic_result", "schema": response_schema},
            }
        response = httpx.post(
            f"{self.api_base}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return ChatResult(
            content=str(choice.get("message", {}).get("content", "")),
            finish_reason=str(choice.get("finish_reason", "")),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )
