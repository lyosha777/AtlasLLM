# RAG baseline evaluation

The default dataset is now **rag-v3.json**, with unchanged questions/passages and
manually curated relevant alternatives. Recall@K uses any-relevant hit semantics.
See [the relevance review](relevance-v3.md) for all seven reviewed failures and labels.

## Ranking failure diagnostics

```powershell
python -m backend.evaluation.rag --strategy dense --failures
```

For each non-gold top-1 result, diagnostics include gold and winning chunk/document
IDs, passage previews, initial/final gold ranks, candidate presence, retrieval scores,
CrossEncoder scores before diversification, and final rankings. Scores are captured
before the reranker mutates candidates and included in saved JSON. Hybrid initial
scores are correctly labeled RRF scores, rather than dense similarity scores.
Summary groups report observed ranking reversals, already-misranked candidates,
missing/omitted gold, same-document competitors, curated distractors, exact duplicate
evidence, and score ties. Semantic equivalence or ambiguous gold requires manual review.

To inspect an existing result without running retrieval:

```powershell
python -m backend.evaluation.rag --results data/benchmarks/YOUR-RAG-RESULT.json --failures
```

Older results support rank/preview analysis but cannot supply scores that were never
recorded. The command says so; it does not rerun inference to fill them in.

## Hybrid retrieval experiment

Compare the same default 45-case dataset with the existing dense-only strategy
and the optional hybrid strategy:

```powershell
python -m backend.evaluation.rag --strategy dense
python -m backend.evaluation.rag --strategy hybrid
```

Production retrieval remains dense-only. The experiment uses BM25 (k1=1.5, b=0.75)
with Unicode word tokenization and case folding over the same Chroma chunks.
Five dense and five lexical results are fused using RRF with rank constant 60,
yielding up to ten unique candidates for the unchanged CrossEncoder and source
diversification. Thus hybrid can increase reranking work compared with dense's
five candidates; compare quality **and** latency. No query expansion or new
chunking is involved. Exact chunk IDs, passages, metadata, and attachment scopes
are retained.

The lexical index is built once per isolated benchmark run; its construction is
included in setup/total time. Retrieval p50 includes dense, BM25, and fusion;
reranker p50 includes the existing reranker. Saved JSON includes strategy, dataset
hash, settings and all existing metrics. Compare runs with matching dataset hashes.
The index is a snapshot; callers reusing it must rebuild after ingestion changes
and may only use it with its original collection and document scope.

From the repository root, with the project environment active:

```powershell
python -m backend.evaluation.rag
```

Or explicitly use the project interpreter:

```powershell
.\.venv\Scripts\python.exe -m backend.evaluation.rag
```

The default versioned `backend/evaluation/datasets/rag-v3.json` contains 65 synthetic
handbook pages and 45 gold questions. The original 25 passages/questions are retained;
20 new questions each have a target passage and a near-topic distractor document.
Challenges include low-overlap paraphrases, acronyms, overloaded terminology,
closely related policies and experimental conditions, conceptual distinctions,
and details within longer contextual passages. Gold document and chunk IDs are
explicit. Pages use Atlas's existing chunking function and PDF-style ID convention;
this evaluates retrieval/reranking, not PDF parsing or generated answers.

JSON results separate the original and challenge cohorts. The original cohort is
now searched against a larger corpus, so its scores are not directly equivalent
to v1. To reproduce the original corpus and questions, run:

```powershell
python -m backend.evaluation.rag --dataset backend/evaluation/datasets/rag-v1.json
```

The harder cases remain explicitly answerable in their gold passages. No score
decrease is assumed: run the benchmark to determine whether these challenges expose
retrieval limitations. Retrieval settings and the separate real-failure register
are unchanged.

The command loads the configured SentenceTransformer and cross-encoder from the
local cache only. It never downloads models, calls Ollama, searches the web, or
opens production SQLite/Chroma. Evaluation Chroma storage lives in a temporary
directory. Only results are written to `data/benchmarks/`.

Expect seconds to tens of seconds on a typical machine, depending on model loading
and CPU/GPU speed. Cold Python/ML imports may take longer. A 120-second hard process
timeout bounds the run; `--timeout 60` changes it. Completed case results are saved
incrementally so a timeout preserves partial measurements. Missing dependencies or
uncached models produce an error rather than an automatic installation.

The printed Recall@1/3/5 and MRR describe **post-reranking exact chunk relevance**.
Recall is success when any curated relevant passage is found, and MRR uses the first relevant rank
(zero when absent). JSON also contains pre-reranking metrics, individual rankings,
questions, gold IDs, model names, dataset hash, timings, and settings. Retrieval uses
the current default five candidates; reranking keeps its current default six-result
limit and document diversification. This can reduce the number of returned passages
when several candidates belong to one document; that behavior is measured unchanged.

Latency uses `time.perf_counter()`. Per-case retrieval and reranking p50 exclude
model loading and indexing; total and setup times include those costs. Source-ID
validity checks returned chunk/document IDs, page, and exact text against the fixture.
It does **not** measure generated citation correctness or claim support.

`failures-v1.json` is a separate real-failure register. The Armenia/Azerbaijan case
is retained verbatim with its observed weak-evidence failure, without a hardcoded
answer or special routing. It is not scored or executed by the local baseline;
reviewed gold evidence is required before objective scoring is possible.

Targeted deterministic validation:

```powershell
python -m unittest backend.tests.test_rag_evaluation -q
```

These tests use fake retrieval/reranking outputs and do not load models. No live
baseline is run automatically by the test suite. Record the first actual result
before making retrieval changes; compare later runs against the same dataset hash
and settings. This fixture is a small controlled baseline, not a claim of general
retrieval quality.
