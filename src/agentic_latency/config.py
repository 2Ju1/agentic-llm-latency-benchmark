from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SweepConfig:
    step_counts: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    tool_tokens: tuple[int, ...] = (64, 256, 512, 1024)
    output_tokens: tuple[int, ...] = (8, 16, 32, 64, 128)
    tool_latency_seconds: tuple[float, ...] = (0, 0.5, 1, 2, 5)
    repeats: int = 3
    fixed_output_repeats: int = 5
    output_repeats: int = 5


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    base_url: str = "http://localhost:8000/v1/chat/completions"
    metrics_urls: dict[str, str] = field(
        default_factory=lambda: {
            "router": "http://localhost:8000/metrics",
            "prefill": "http://localhost:8100/metrics",
            "decode": "http://localhost:8201/metrics",
        }
    )
    temperature: float = 0
    warmup: bool = True
    default_num_steps: int = 8
    default_tool_tokens: int = 256
    default_max_output_tokens: int = 32
    sweeps: SweepConfig = field(default_factory=SweepConfig)


def _as_tuple(data: dict[str, Any], key: str, default: tuple[Any, ...]) -> tuple[Any, ...]:
    value = data.get(key, default)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"sweeps.{key} must be a list")
    return tuple(value)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file) or {}

    sweep_raw = raw.get("sweeps", {})
    defaults = SweepConfig()
    sweeps = SweepConfig(
        step_counts=_as_tuple(sweep_raw, "step_counts", defaults.step_counts),
        tool_tokens=_as_tuple(sweep_raw, "tool_tokens", defaults.tool_tokens),
        output_tokens=_as_tuple(sweep_raw, "output_tokens", defaults.output_tokens),
        tool_latency_seconds=_as_tuple(
            sweep_raw, "tool_latency_seconds", defaults.tool_latency_seconds
        ),
        repeats=int(sweep_raw.get("repeats", defaults.repeats)),
        fixed_output_repeats=int(
            sweep_raw.get("fixed_output_repeats", defaults.fixed_output_repeats)
        ),
        output_repeats=int(sweep_raw.get("output_repeats", defaults.output_repeats)),
    )

    defaults_config = ExperimentConfig()
    metrics_urls = dict(defaults_config.metrics_urls)
    metrics_urls.update(raw.get("metrics_urls", {}))
    base_url = os.getenv(
        "AGENTIC_LATENCY_BASE_URL", raw.get("base_url", defaults_config.base_url)
    )
    model = os.getenv("AGENTIC_LATENCY_MODEL", raw.get("model", defaults_config.model))

    return ExperimentConfig(
        model=model,
        base_url=base_url,
        metrics_urls=metrics_urls,
        temperature=float(raw.get("temperature", 0)),
        warmup=bool(raw.get("warmup", True)),
        default_num_steps=int(raw.get("default_num_steps", 8)),
        default_tool_tokens=int(raw.get("default_tool_tokens", 256)),
        default_max_output_tokens=int(raw.get("default_max_output_tokens", 32)),
        sweeps=sweeps,
    )

