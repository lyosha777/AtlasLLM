# Atlas Performance Benchmark — Baseline v1

This benchmark establishes the initial performance baseline for Atlas before
performance-specific optimization.

## Configuration

| Setting | Value |
|---|---|
| Model | `qwen3:4b-instruct` |
| Inference provider | Ollama |
| Deployment | Local |
| Benchmark mode | Quick |
| Completed runs | 3 / 3 |
| Maximum output | 128 tokens |
| Web retrieval | Disabled |
| Deep Research | Disabled |

## Results

| Metric | Median | Mean |
|---|---:|---:|
| User-perceived TTFT | 1,411.4 ms | 2,635.3 ms |
| Inference TTFT | 88.6 ms | 90.1 ms |
| Throughput (TPS) | 95.1 tok/s | 95.1 tok/s |
| TPOT | 10.5 ms/token | 10.7 ms/token |
| End-to-end latency | 1.579 s | 2.718 s |
| Planner latency | 503 ms | 1,698.3 ms |

Total benchmark runtime: **16.8 seconds**.

## What the metrics mean

**User-perceived TTFT (Time To First Token)** measures the time between Atlas
accepting a message and the first visible response content arriving through
the streaming API.

**Inference TTFT** measures the time between starting the Ollama inference
request and receiving the first generated output from the model.

**TPS (Tokens Per Second)** measures model generation throughput using
Ollama's reported output token count and evaluation duration.

**TPOT (Time Per Output Token)** measures the average model decode time for
each generated output token.

**E2E (End-to-End) latency** measures the total time between submitting the
request and completing the response.

**Planner latency** measures the time spent deciding whether Atlas should use
documents, web retrieval, freshness-sensitive search, or ordinary chat.

## Initial observations

The model itself begins inference quickly.

Median inference TTFT was approximately **89 ms**, while median
user-perceived TTFT was approximately **1.41 seconds**.

This indicates that most perceived latency occurs outside the model's initial
inference latency.

The planner is one contributor to this overhead. Its median latency was
**503 ms**.

The first benchmark request was a significant outlier:

- Planner: 4.107 s
- User TTFT: 5.100 s
- E2E: 5.101 s

The following two requests had user TTFT values of approximately 1.41 s and
1.39 s respectively, suggesting that startup/cold-path behavior warrants
further investigation.

No optimization conclusions are drawn from this benchmark yet. This run is
intended to establish a baseline.

## Methodology

The Quick benchmark uses three short controlled prompts and is intended for
fast development feedback rather than statistically rigorous performance
analysis.

Timing uses high-resolution monotonic measurements.

TPS and TPOT are calculated from Ollama's reported `eval_count` and
`eval_duration` rather than approximating tokens from words or characters.

Atlas distinguishes user-perceived TTFT from inference TTFT so application
overhead is not incorrectly attributed to model inference.

The current Ollama streaming interface does not provide sufficiently reliable
individual token boundaries for Atlas to claim true token-level ITL
(Inter-Token Latency).

Stream-chunk intervals are therefore recorded separately and are not reported
as token ITL.

## Limitations

This baseline contains only three runs.

It is useful for development profiling, but is too small for statistically
meaningful tail-latency measurements. In particular, p95 values from this
Quick benchmark should not be interpreted as production-quality p95
estimates.

A larger Full benchmark will be performed when statistically meaningful
performance comparisons are required.

Document retrieval, reranking, web research, and Deep Research were not
exercised in this baseline and therefore have no latency measurements in this
report.

## Next steps

Future benchmark versions will compare this baseline against optimized Atlas
versions and additional workloads, including:

- document RAG
- retrieval and reranking
- web-assisted responses
- Deep Research
- alternative local models
- retrieval strategy improvements

Performance changes will be reported relative to this baseline rather than
estimated subjectively.