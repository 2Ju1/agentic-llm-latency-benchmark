# Experiment design

## Question

Agentic system은 매 step마다 tool result와 대화 기록을 prompt에 누적한다. 이 실험은
누적된 context가 TTFT와 end-to-end latency에 주는 영향을 다음 요소로 분리한다.

1. prompt/context length
2. tool result size
3. output length
4. stable prefix와 front variation
5. external tool latency

## Controlled sweeps

| Experiment | Changed factor | Primary metric | Controlled factors |
|---|---|---|---|
| `vary_steps_fixed_output` | step count | TTFT | tool tokens, output tokens, prefix |
| `vary_tool_tokens` | tool result size | TTFT | steps, output tokens, prefix |
| `prefix_stability` | first-prefix stability | cache hits, TTFT | steps, tool tokens, output tokens |
| `vary_output_tokens` | max output tokens | LLM latency | steps, tool tokens, prefix |
| `vary_tool_latency` | synthetic tool delay | E2E latency | prompt and output configuration |

Default values and sweep ranges live in `configs/default.yaml`; the CLI renders a deterministic
request plan from this file before any network request is made.

## Measurement boundaries

- **TTFT:** client request start to the first non-empty streamed content token.
- **LLM latency:** client request start to the end of the SSE stream.
- **E2E latency:** synthetic tool delay plus LLM latency.
- **Metric delta:** Prometheus counter after the request minus the value before the request.
- **Counter reset:** if the after value is lower, the delta is recorded as missing rather than a
  negative amount of work.

The measured Prometheus metrics include prompt/generation tokens, local compute tokens, local
cache-hit tokens, external KV transfer tokens, prefill time, decode time, queue time, and TTFT.

## Workload

The base conversation asks the model to plan a three-day trip. For `step=N`, the generator adds
`N-1` assistant/tool-result pairs. `tool_tokens` controls repeated content inside each result. A
request identifier is inserted at the front only for the `front_variation` condition, invalidating
the otherwise stable prefix.

## Provenance

Curated data under `results/` was measured on 2026-06-10 with:

- Qwen/Qwen2.5-1.5B-Instruct
- LLM-D + vLLM
- 2 × RTX 3090
- temperature 0
- concurrency 1

The raw CSV contains one row per measured request. `summary.csv` is grouped by experiment and
condition. Figures in the repository were generated from this evidence.

## Limitations

- The benchmark is synthetic and isolates mechanisms; it is not a production traffic trace.
- Results are hardware- and model-specific.
- Concurrency is fixed to one, so the study does not characterize queueing under load.
- The first-token attention explanation is supported by cache/prefill/transfer metrics but is not
  a kernel-level causal attribution.
- Tool latency is simulated with a delay; real tools have different variance and failure modes.

