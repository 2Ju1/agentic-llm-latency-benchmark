from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class CompletionResult:
    output_text: str
    ttft_sec: float | None
    total_latency_sec: float
    error: str = ""


def extract_sse_content(lines: Iterable[bytes]) -> Iterable[str]:
    for line in lines:
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        data = decoded.removeprefix("data: ").strip()
        if data == "[DONE]":
            return
        payload = json.loads(data)
        content = payload["choices"][0].get("delta", {}).get("content", "")
        if content:
            yield content


def stream_chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[Mapping[str, str]],
    max_tokens: int,
    temperature: float = 0,
    timeout: float = 120,
) -> CompletionResult:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    start = time.perf_counter()
    first_token_time: float | None = None
    output: list[str] = []

    try:
        with requests.post(
            base_url,
            json=payload,
            stream=True,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            for content in extract_sse_content(response.iter_lines()):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                output.append(content)
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as exc:
        return CompletionResult(
            output_text="".join(output),
            ttft_sec=None if first_token_time is None else first_token_time - start,
            total_latency_sec=time.perf_counter() - start,
            error=str(exc),
        )

    end = time.perf_counter()
    return CompletionResult(
        output_text="".join(output),
        ttft_sec=None if first_token_time is None else first_token_time - start,
        total_latency_sec=end - start,
    )

