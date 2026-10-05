# Evaluating Atlas

Atlas is a local-first research assistant with document retrieval, reranking,
cited answers, and background generation. Its evaluation separates model speed,
retrieval coverage, and final source ranking. The results below come from saved
local runs, not estimated performance or new experiments.

## Inference and perceived latency

The [quick inference baseline](benchmarks/raw/inference-baseline.json) used
Ollama `qwen3:4b-instruct` for three short prompts: greeting, arithmetic, and
rewriting. It included Atlas's normal planner, with no web or Deep Research.

| Metric | Median | Mean |
| --- | ---: | ---: |
| API-visible TTFT | 1,411.4 ms | 2,635.3 ms |
| Inference TTFT | 88.6 ms | 90.1 ms |
| Generation throughput | 95.1 tokens/s | 95.1 tokens/s |
| TPOT, decode average | 10.5 ms/token | 10.7 ms/token |
| End-to-end response latency | 1.579 s | 2.718 s |
| Planner latency | 503 ms | 1,698.3 ms |

**API-visible TTFT** measures submission to the first nonempty answer content
observed by the benchmark's SSE client. It approximates the user's wait for an
answer, including orchestration and streaming delivery; it does **not** measure
browser rendering or paint. **Inference TTFT** measures the provider request to
the first filtered answer chunk, excluding private reasoning content. E2E ends
when the client observes the terminal job state. Timing uses `time.perf_counter()`.

TPS and TPOT use Ollama's output-token counts and decode duration, not word counts.
TPOT is an average decode cost, not measured token-to-token arrival spacing.
Stream chunks are not assumed to represent individual tokens.

Inference started producing answer content quickly, but the complete application
path took longer. Planning contributed materially: the first prompt spent
4.107 s in the planner and reached visible content after 5.100 s. Other
orchestration/delivery costs also remain in TTFT; the planner does not explain
the whole gap. This is a small diagnostic baseline: outputs contained only
3, 3, and 13 tokens, so throughput is not a sustained-generation claim.

## Frozen document RAG result

The [final curated run](benchmarks/raw/rag-final-v2.json) completed all **45 cases**
against **65 controlled local passages**. It uses `all-MiniLM-L6-v2` embeddings,
dense retrieval, and `cross-encoder/ms-marco-MiniLM-L6-v2` reranking with the
existing document-diversification behavior.

| Metric | Result |
| --- | ---: |
| Recall@1 | 0.933 (42/45) |
| Recall@3 | 1.000 |
| Recall@5 | 1.000 |
| MRR | 0.967 |
| Source-ID validity | 100% |
| Retrieval p50 | 35.24 ms |
| Reranker p50 | 140.11 ms |
| Total benchmark time | 38.96 s |

For this benchmark, **Recall@K is an any-relevant hit rate**: a query succeeds
when at least one manually curated relevant chunk appears in its first K results.
**MRR** averages the reciprocal rank of the first relevant chunk, with zero for
a miss. The table describes final ranking after reranking.

Source-ID validity checks chunk/document identity, page, and exact passage against
the corpus. It does not measure answer accuracy, citation entailment, or source
trustworthiness. This benchmark generates no LLM answers and uses no live web.
Per-stage latencies exclude model loading and corpus indexing; total includes
setup and process overhead. Evaluation Chroma storage is temporary and separate
from production documents, Chroma, and SQLite.

## How the ground truth evolved

1. The [initial 25-query corpus](benchmarks/raw/rag-initial-25.json) scored 1.000
   on Recall@1/3/5 and MRR. Its distinct topics made it insufficiently discriminative.
2. The corpus expanded to 45 questions with low-overlap paraphrases, acronyms,
   terminology ambiguity, closely related facts, longer contextual passages, and
   near-topic distractor documents. The [first complete harder run](benchmarks/raw/rag-baseline-v1.json)
   reached Recall@1 0.844 and MRR 0.915, while Recall@3/5 remained 1.000.
3. Ranking diagnostics exposed labeling problems: some noncanonical winners
   independently contained the required evidence. Manual passage review added
   four relevant alternatives, retaining one canonical gold chunk per question.
   No LLM, embedding score, or reranker score assigned relevance.
4. Incomplete and wrong-context evidence remained non-relevant: naming MRR without
   its reciprocal rule, giving RTO instead of the requested RPO, and describing
   the earlier experiment instead of its follow-up. The [review decisions](benchmarks/relevance-v3.md)
   document all seven original top-1 misses.

The final labels are frozen in [rag-v3.json](../backend/evaluation/datasets/rag-v3.json).
The corpus and retrieval were unchanged during label review; the higher final
score is a ground-truth correction, not an algorithmic improvement. Historical
dataset versions remain available. A separate [real-failure register](../backend/evaluation/datasets/failures-v1.json)
preserves the Armenia/Azerbaijan weak-evidence case, unscored until reviewed gold
evidence is available.

## Hybrid retrieval experiment: not adopted

Before relevance-label correction, dense retrieval was compared with dense +
BM25, fused by Reciprocal Rank Fusion (RRF), followed by the same CrossEncoder.
Both runs used the same 45-query dataset and then-current labels.

| Final-ranking metric | Dense + CrossEncoder | Hybrid + CrossEncoder |
| --- | ---: | ---: |
| Recall@1 | 0.844 | 0.844 |
| Recall@3 / Recall@5 | 1.000 / 1.000 | 1.000 / 1.000 |
| MRR | 0.915 | 0.915 |
| Retrieval p50 | 36.5 ms | 37.9 ms |
| Reranker p50 | 142.4 ms | 189.9 ms |

Hybrid supplied up to ten fused candidates rather than dense's five. It increased
reranking latency by about 33% without improving final quality on this benchmark.
It remains an optional evaluation strategy; production retrieval stays dense-only.
Additional architectural complexity was not treated as evidence of improvement.
These historical scores should not be compared directly with corrected-label scores.

## Engineering lessons and limits

- Perfect scores can expose an easy evaluation rather than a solved retrieval problem.
- Candidate retrieval and final ranking are different problems. Relevant evidence
  appeared within the top three even when a distractor won first place.
- More complex retrieval does not automatically produce better results; the hybrid
  experiment added measurable cost without a final-ranking benefit.
- Quality and latency need joint measurement, with candidate budgets recorded.
- Evaluation itself needs validation: canonical source identity is not the same
  as relevance when another passage independently answers the question.

These are single-machine, small-corpus measurements, not broad production claims.
Hardware and repeated-run confidence intervals were not captured in these artifacts.
The controlled benchmark does not cover PDF extraction, live web research, or
generated-answer quality.

## Reproduce and inspect

From the repository root with the project environment active:

```powershell
python -m backend.evaluation.benchmark
python -m backend.evaluation.rag --strategy dense --failures
```

See the [harness guide](benchmarks/rag-baseline.md) for isolation, timeouts, optional
hybrid comparison, and inspecting saved results without rerunning inference.

### Raw artifacts

Copies preserve the original generated JSON contents. All runs below completed;
the failed partial Windows-cleanup run is not used as a baseline.

| Stable artifact | Original generated filename |
| --- | --- |
| [Inference baseline](benchmarks/raw/inference-baseline.json) | `20261005T000531Z-0149f937.json` |
| [Initial 25-case RAG](benchmarks/raw/rag-initial-25.json) | `20261005T003154Z-rag-cbf18fae.json` |
| [Harder baseline v1](benchmarks/raw/rag-baseline-v1.json) | `20261005T004011Z-rag-b095fee5.json` |
| [Comparison: dense](benchmarks/raw/rag-dense-vs-hybrid-dense.json) | `20261005T004625Z-rag-b4ca1138.json` |
| [Comparison: hybrid](benchmarks/raw/rag-dense-vs-hybrid-hybrid.json) | `20261005T004647Z-rag-e7e29f91.json` |
| [Final curated baseline v2](benchmarks/raw/rag-final-v2.json) | `20261005T011351Z-rag-deb88194.json` |

The final artifact's filename names the baseline revision; its internal dataset
version is `rag-v3`. Dataset hashes and model/settings metadata remain in the JSON.
