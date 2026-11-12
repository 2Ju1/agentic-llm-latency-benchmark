from __future__ import annotations

import re
from collections.abc import Mapping

import requests

MetricValues = dict[str, float | None]

METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s+\d+)?$"
)
LABEL_RE = re.compile(
    r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"(?P<value>(?:\\.|[^"\\])*)"'
)

TRACKED_METRICS = {
    "prompt_tokens": ("vllm:prompt_tokens_total", None),
    "generation_tokens": ("vllm:generation_tokens_total", None),
    "local_compute_tokens": (
        "vllm:prompt_tokens_by_source_total",
        {"source": "local_compute"},
    ),
    "local_cache_hit_tokens": (
        "vllm:prompt_tokens_by_source_total",
        {"source": "local_cache_hit"},
    ),
    "external_kv_transfer_tokens": (
        "vllm:prompt_tokens_by_source_total",
        {"source": "external_kv_transfer"},
    ),
    "prompt_tokens_cached": ("vllm:prompt_tokens_cached_total", None),
    "prompt_tokens_recomputed": ("vllm:prompt_tokens_recomputed_total", None),
    "prefill_time_sec": ("vllm:request_prefill_time_seconds_sum", None),
    "decode_time_sec": ("vllm:request_decode_time_seconds_sum", None),
    "queue_time_sec": ("vllm:request_queue_time_seconds_sum", None),
    "ttft_sec": ("vllm:time_to_first_token_seconds_sum", None),
}


def parse_labels(label_text: str | None) -> dict[str, str]:
    if not label_text:
        return {}
    return {
        match.group("key"): bytes(match.group("value"), "utf-8").decode("unicode_escape")
        for match in LABEL_RE.finditer(label_text)
    }


def parse_metrics_text(text: str) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE_RE.match(line)
        if match is None:
            continue
        samples.append(
            {
                "name": match.group("name"),
                "labels": parse_labels(match.group("labels")),
                "value": float(match.group("value")),
            }
        )
    return samples


def find_metric(
    samples: list[dict[str, object]],
    name: str,
    labels: Mapping[str, str] | None = None,
) -> float | None:
    expected_labels = labels or {}
    for sample in samples:
        if sample["name"] != name:
            continue
        sample_labels = sample["labels"]
        if not isinstance(sample_labels, dict):
            continue
        if all(sample_labels.get(key) == value for key, value in expected_labels.items()):
            return float(sample["value"])
    return None


def collect_metrics(metrics_url: str, timeout: float = 10) -> MetricValues:
    response = requests.get(metrics_url, timeout=timeout)
    response.raise_for_status()
    samples = parse_metrics_text(response.text)
    return {
        alias: find_metric(samples, name, labels)
        for alias, (name, labels) in TRACKED_METRICS.items()
    }


def collect_metrics_by_source(metrics_urls: Mapping[str, str]) -> dict[str, MetricValues]:
    collected: dict[str, MetricValues] = {}
    for source, url in metrics_urls.items():
        try:
            collected[source] = collect_metrics(url)
        except requests.RequestException:
            collected[source] = {}
    return collected


def diff_metrics(before: MetricValues, after: MetricValues) -> MetricValues:
    delta: MetricValues = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if before_value is None or after_value is None or after_value < before_value:
            delta[key] = None
        else:
            delta[key] = after_value - before_value
    return delta


def flatten_metric_deltas(
    before: Mapping[str, MetricValues],
    after: Mapping[str, MetricValues],
) -> MetricValues:
    flattened: MetricValues = {}
    for source, after_values in after.items():
        for name, value in diff_metrics(before.get(source, {}), after_values).items():
            flattened[f"{source}_{name}_delta"] = value
    return flattened

