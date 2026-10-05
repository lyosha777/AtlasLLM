# Curated relevance ground truth v3

The 45 questions and 65 passage texts are unchanged from v2. Each case now retains
`primary_gold_chunk_id` for canonical diagnostics and `relevant_chunk_ids` for metrics.
Legacy `expected_chunk_ids` and canonical document IDs remain compatible.

Recall@K is an any-relevant hit rate: a case succeeds if at least one curated
relevant passage occurs in the first K results. MRR uses the first relevant rank.
This differs from the previous fraction-of-gold-passages definition; compare runs
using the same labels and metric semantics. Source-ID validity is unchanged.
`Relevant alternatives` counts curated IDs beyond each case's primary gold.

Manual review of the seven saved top-1 failures:

| Case | Winner review | Decision |
| --- | --- | --- |
| q04 | JWT-related TLS passage explicitly explains certificate chain and hostname verification of server identity. | Relevant alternative |
| q14 | Search-metrics passage names MRR and first-result position but omits the reciprocal formula needed for the numeric answer. | Non-relevant; incomplete evidence |
| q19 | SSE-related passage explicitly supplies Last-Event-ID on reconnect. | Relevant alternative |
| q22 | Cedar-related passage includes both pH buffers and replacement drift threshold. | Relevant alternative |
| q24 | Archive passage explicitly prohibits returning thawed primary samples and separately qualifies working aliquots. | Relevant alternative |
| q26 | Recovery-related passage gives RTO downtime, not the required RPO data-loss limit. | Non-relevant; wrong concept |
| q42 | Original germination passage reverses the follow-up trial's lighting assignment. | Non-relevant; wrong experiment |

Four alternatives were curated from the actual passage text, without an LLM or
score-based inference. Near-topic passages are not automatically relevant. v1/v2
files remain available for reproducing historical results. The default benchmark
uses v3; corpus, retrieval, reranking, and source validation are unchanged.

```powershell
python -m backend.evaluation.rag --strategy dense --failures
```

Diagnostics distinguish a canonical gold displacement by a curated relevant
alternative from a genuine top-1 miss. Historical result hashes deliberately do
not match the new labels; rerun explicitly to obtain the new metric baseline.
