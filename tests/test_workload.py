import unittest

from agentic_latency.workload import build_agentic_messages


class WorkloadTest(unittest.TestCase):
    def test_step_count_controls_accumulated_tool_results(self) -> None:
        messages = build_agentic_messages(step=4, tool_tokens=2)

        self.assertEqual(len(messages), 8)
        self.assertTrue(messages[-1]["content"].startswith("[TOOL_RESULT_3]"))
        self.assertEqual(messages[-1]["content"].count("hotel"), 2)

    def test_request_id_changes_the_front_prefix(self) -> None:
        stable = build_agentic_messages(step=1)
        varied = build_agentic_messages(step=1, request_id="run-1")

        self.assertNotEqual(stable[0]["content"], varied[0]["content"])
        self.assertTrue(varied[0]["content"].startswith("[REQUEST_ID=run-1]"))

    def test_invalid_step_is_rejected(self) -> None:
        for step in (0, -1):
            with self.subTest(step=step), self.assertRaisesRegex(ValueError, "step"):
                build_agentic_messages(step=step)


if __name__ == "__main__":
    unittest.main()

