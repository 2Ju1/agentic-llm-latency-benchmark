# Result index

| Artifact | Description |
|---|---|
| `results_raw.csv` | 111 measured requests with client and vLLM metric deltas |
| `summary.csv` | Aggregation by experiment and controlled condition |

The files are curated evidence, not runtime output. New runs are written to the ignored
`outputs/` directory by default so a local experiment does not silently overwrite the reported
data.

## Primary comparisons

- `vary_steps_fixed_output`: context accumulation with output fixed to 32 tokens
- `vary_tool_tokens`: tool-result size sweep
- `prefix_stability`: stable prefix versus request-specific front variation
- `vary_output_tokens`: decode-length sweep
- `vary_tool_latency`: separation of LLM and end-to-end latency

See [`docs/experiment-design.md`](../docs/experiment-design.md) for measurement boundaries,
hardware, controls, and known limitations.

