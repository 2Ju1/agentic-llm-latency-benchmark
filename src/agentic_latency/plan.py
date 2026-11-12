from __future__ import annotations

from dataclasses import asdict, dataclass

from agentic_latency.config import ExperimentConfig


@dataclass(frozen=True)
class ExperimentCase:
    experiment: str
    condition: str
    run_id: int
    num_steps: int
    tool_tokens: int
    max_output_tokens: int
    target_output_tokens: int | None = None
    tool_latency_sec: float = 0
    prefix_condition: str = "stable"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_plan(config: ExperimentConfig) -> list[ExperimentCase]:
    plan: list[ExperimentCase] = []
    sweeps = config.sweeps

    for num_steps in sweeps.step_counts:
        for run_id in range(1, sweeps.fixed_output_repeats + 1):
            plan.append(
                ExperimentCase(
                    experiment="vary_steps_fixed_output",
                    condition=f"num_steps={num_steps}",
                    run_id=run_id,
                    num_steps=num_steps,
                    tool_tokens=config.default_tool_tokens,
                    max_output_tokens=config.default_max_output_tokens,
                    target_output_tokens=config.default_max_output_tokens,
                )
            )

    for tool_tokens in sweeps.tool_tokens:
        for run_id in range(1, sweeps.repeats + 1):
            plan.append(
                ExperimentCase(
                    experiment="vary_tool_tokens",
                    condition=f"tool_tokens={tool_tokens}",
                    run_id=run_id,
                    num_steps=config.default_num_steps,
                    tool_tokens=tool_tokens,
                    max_output_tokens=config.default_max_output_tokens,
                )
            )

    for prefix_condition in ("stable", "front_variation"):
        for run_id in range(1, sweeps.repeats + 1):
            plan.append(
                ExperimentCase(
                    experiment="prefix_stability",
                    condition=prefix_condition,
                    run_id=run_id,
                    num_steps=config.default_num_steps,
                    tool_tokens=config.default_tool_tokens,
                    max_output_tokens=config.default_max_output_tokens,
                    prefix_condition=prefix_condition,
                )
            )

    for output_tokens in sweeps.output_tokens:
        for run_id in range(1, sweeps.output_repeats + 1):
            plan.append(
                ExperimentCase(
                    experiment="vary_output_tokens",
                    condition=f"max_output_tokens={output_tokens}",
                    run_id=run_id,
                    num_steps=config.default_num_steps,
                    tool_tokens=config.default_tool_tokens,
                    max_output_tokens=output_tokens,
                    target_output_tokens=output_tokens,
                )
            )

    for tool_latency_sec in sweeps.tool_latency_seconds:
        for run_id in range(1, sweeps.repeats + 1):
            plan.append(
                ExperimentCase(
                    experiment="vary_tool_latency",
                    condition=f"tool_latency_sec={tool_latency_sec}",
                    run_id=run_id,
                    num_steps=config.default_num_steps,
                    tool_tokens=config.default_tool_tokens,
                    max_output_tokens=config.default_max_output_tokens,
                    tool_latency_sec=tool_latency_sec,
                )
            )
    return plan

