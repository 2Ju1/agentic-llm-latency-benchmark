import unittest

from agentic_latency.plan import ExperimentCase
from agentic_latency.runner import make_messages, output_length_instruction


class RunnerTest(unittest.TestCase):
    def test_output_instruction_is_only_added_for_controlled_output_length(self) -> None:
        self.assertIsNone(output_length_instruction(None))
        instruction = output_length_instruction(32)
        self.assertIsNotNone(instruction)
        self.assertIn("32 output tokens", instruction)

    def test_front_variation_changes_the_first_message(self) -> None:
        stable = ExperimentCase("prefix_stability", "stable", 1, 2, 4, 8)
        varied = ExperimentCase(
            "prefix_stability",
            "front_variation",
            1,
            2,
            4,
            8,
            prefix_condition="front_variation",
        )

        self.assertTrue(make_messages(stable)[0]["content"].startswith("You are"))
        self.assertTrue(make_messages(varied)[0]["content"].startswith("[REQUEST_ID="))


if __name__ == "__main__":
    unittest.main()

