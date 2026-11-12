from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from agentic_latency.client import stream_chat_completion
from agentic_latency.config import ExperimentConfig
from agentic_latency.metrics import collect_metrics_by_source, flatten_metric_deltas
from agentic_latency.plan import ExperimentCase, build_plan
from agentic_latency.tokenizer import TokenCounter
from agentic_latency.workload import build_agentic_messages


def output_length_instruction(target_output_tokens: int | None) -> str | None:
    if target_output_tokens is None:
        return None
    return (
        "For the final answer, write one coherent paragraph close to "
        f"{target_output_tokens} output tokens. Use the available output budget."
    )


def make_messages(case: ExperimentCase) -> list[dict[str, str]]:
    request_id = None
    if case.prefix_condition == "front_variation":
        request_id = f"{case.experiment}-{case.condition}-{case.run_id}-{uuid.uuid4()}"
    return build_agentic_messages(
        step=case.num_steps,
        tool_tokens=case.tool_tokens,
        request_id=request_id,
        final_instruction=output_length_instruction(case.target_output_tokens),
    )


def run_case(
    config: ExperimentConfig,
    case: ExperimentCase,
    token_counter: TokenCounter,
) -> dict[str, object]:
    messages = make_messages(case)
    prompt_tokens = token_counter.count_prompt(messages)
    before_metrics = collect_metrics_by_source(config.metrics_urls)

    started = time.perf_counter()
    if case.tool_latency_sec > 0:
        time.sleep(case.tool_latency_sec)
    completion = stream_chat_completion(
        base_url=config.base_url,
        model=config.model,
        messages=messages,
        max_tokens=case.max_output_tokens,
        temperature=config.temperature,
    )
    end_to_end_latency = time.perf_counter() - started
    after_metrics = collect_metrics_by_source(config.metrics_urls)

    row: dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **case.to_dict(),
        "prompt_tokens": prompt_tokens,
        "output_tokens": token_counter.count_output(completion.output_text),
        "ttft_sec": completion.ttft_sec,
        "llm_latency_sec": completion.total_latency_sec,
        "end_to_end_latency_sec": end_to_end_latency,
        "error": completion.error,
    }
    row.update(flatten_metric_deltas(before_metrics, after_metrics))
    return row


def summarize_results(raw_results: Path, summary_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(raw_results)
    ok = frame[frame["error"].fillna("") == ""].copy()
    grouping = [
        "experiment",
        "condition",
        "num_steps",
        "tool_tokens",
        "max_output_tokens",
        "target_output_tokens",
        "tool_latency_sec",
        "prefix_condition",
    ]
    summary = (
        ok.groupby(grouping, dropna=False)
        .agg(
            runs=("run_id", "count"),
            prompt_tokens_mean=("prompt_tokens", "mean"),
            output_tokens_mean=("output_tokens", "mean"),
            ttft_sec_mean=("ttft_sec", "mean"),
            ttft_sec_std=("ttft_sec", "std"),
            llm_latency_sec_mean=("llm_latency_sec", "mean"),
            end_to_end_latency_sec_mean=("end_to_end_latency_sec", "mean"),
        )
        .reset_index()
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return summary


def execute_plan(
    config: ExperimentConfig,
    output_path: Path,
    *,
    warmup: bool | None = None,
) -> pd.DataFrame:
    plan = build_plan(config)
    token_counter = TokenCounter(config.model)
    use_warmup = config.warmup if warmup is None else warmup
    warmed_conditions: set[tuple[str, str]] = set()
    rows: list[dict[str, object]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for case in plan:
        condition_key = (case.experiment, case.condition)
        if use_warmup and condition_key not in warmed_conditions:
            run_case(config, case, token_counter)
            warmed_conditions.add(condition_key)

        row = run_case(config, case, token_counter)
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(
            f"{case.experiment} {case.condition} run={case.run_id} "
            f"ttft={row['ttft_sec']} error={row['error']}"
        )

    return pd.DataFrame(rows)

