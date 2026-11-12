import unittest

from agentic_latency.config import ExperimentConfig, SweepConfig
from agentic_latency.plan import build_plan


class PlanTest(unittest.TestCase):
    def test_plan_is_deterministic_and_covers_each_controlled_sweep(self) -> None:
        config = ExperimentConfig(
            sweeps=SweepConfig(
                step_counts=(1, 2),
                tool_tokens=(64,),
                output_tokens=(8,),
                tool_latency_seconds=(0,),
                repeats=2,
                fixed_output_repeats=2,
                output_repeats=2,
            )
        )

        plan = build_plan(config)

        self.assertEqual(len(plan), 14)
        self.assertEqual(plan, build_plan(config))
        self.assertEqual(
            {case.experiment for case in plan},
            {
                "vary_steps_fixed_output",
                "vary_tool_tokens",
                "prefix_stability",
                "vary_output_tokens",
                "vary_tool_latency",
            },
        )


if __name__ == "__main__":
    unittest.main()

