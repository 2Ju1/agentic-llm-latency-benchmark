import unittest

from agentic_latency.metrics import diff_metrics, find_metric, parse_metrics_text


METRICS = """
# HELP vllm:prompt_tokens_total Total prompt tokens
vllm:prompt_tokens_total 120
vllm:prompt_tokens_by_source_total{source="local_compute"} 20
vllm:prompt_tokens_by_source_total{source="local_cache_hit"} 100
vllm:request_prefill_time_seconds_sum 1.25 1710000000
"""


class MetricsTest(unittest.TestCase):
    def test_prometheus_metrics_are_parsed_with_labels_and_timestamp(self) -> None:
        samples = parse_metrics_text(METRICS)

        self.assertEqual(find_metric(samples, "vllm:prompt_tokens_total"), 120)
        self.assertEqual(
            find_metric(
                samples,
                "vllm:prompt_tokens_by_source_total",
                {"source": "local_cache_hit"},
            ),
            100,
        )
        self.assertEqual(
            find_metric(samples, "vllm:request_prefill_time_seconds_sum"), 1.25
        )

    def test_counter_reset_is_not_reported_as_negative_work(self) -> None:
        delta = diff_metrics(
            {"tokens": 100, "latency": 1.0},
            {"tokens": 10, "latency": 1.4},
        )

        self.assertIsNone(delta["tokens"])
        self.assertAlmostEqual(delta["latency"], 0.4)


if __name__ == "__main__":
    unittest.main()

