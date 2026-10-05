"""Offline baseline for the existing document retrieval and reranking pipeline."""
import argparse
import hashlib
import json
import multiprocessing
import shutil
from contextlib import nullcontext
from pathlib import Path
import statistics
import tempfile
import time
from datetime import datetime, timezone
from uuid import uuid4
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(__file__).with_name("datasets") / "rag-v3.json"


def load_dataset(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    passages = {item["id"]: item for item in data["passages"]}
    if len(passages) != len(data["passages"]):
        raise ValueError("Duplicate passage IDs")
    case_ids = set()
    for case in data["cases"]:
        if case["id"] in case_ids or not case["question"].strip():
            raise ValueError("Duplicate case ID or empty question")
        case_ids.add(case["id"])
        if not case["expected_chunk_ids"] or not set(case["expected_chunk_ids"]) <= passages.keys():
            raise ValueError("Gold case references an unknown passage")
        if {passages[item]["document_id"] for item in case["expected_chunk_ids"]} != set(case["expected_document_ids"]):
            raise ValueError("Gold document and passage IDs disagree")
        relevant = case.get("relevant_chunk_ids", case["expected_chunk_ids"])
        primary = case.get("primary_gold_chunk_id", case["expected_chunk_ids"][0])
        if not relevant or len(set(relevant)) != len(relevant) or not set(relevant) <= passages.keys() or primary not in relevant or primary not in case["expected_chunk_ids"]:
            raise ValueError("Invalid curated relevant IDs or canonical primary gold")
    return data


def ranking_metrics(ids, expected):
    expected = set(expected)
    return {**{f"recall_at_{k}": float(bool(set(ids[:k]) & expected)) for k in (1, 3, 5)},
            "mrr": next((1 / rank for rank, item in enumerate(ids, 1) if item in expected), 0.0)}


def source_valid(source, passages):
    gold = passages.get(source.chunk_id)
    return bool(gold and source.document_id == gold["document_id"] and source.text == gold["text"]
                and source.page == gold["page"] and source.source_type == "document")


def ranking_snapshot(sources):
    return [dict(rank=rank, chunk_id=item.chunk_id, document_id=item.document_id, title=item.title,
                 page=item.page, score=item.relevance_score, preview=" ".join(item.text.split())[:240])
            for rank, item in enumerate(sources, 1)]


def failure_analysis(records, data):
    passages = {item["id"]: item for item in data["passages"]}
    cases = {item["id"]: item for item in data["cases"]}
    failures, patterns = [], Counter()
    for record in records:
        case = cases[record["case_id"]]
        primary = case.get("primary_gold_chunk_id", case["expected_chunk_ids"][0])
        expected = case.get("relevant_chunk_ids", case["expected_chunk_ids"])
        final_ids, initial_ids = record["ranked_ids"], record["retrieved_ids"]
        if final_ids and final_ids[0] == primary:
            continue
        alternative_won = bool(final_ids and final_ids[0] in expected)
        tags = []
        present = any(item in initial_ids for item in expected)
        if not present:
            tags.append("expected evidence absent from initial candidates")
        elif initial_ids and initial_ids[0] in expected:
            tags.append("initial retrieval ranked correctly; final ranking reversed it")
        else:
            tags.append("initial retrieval already ranked a non-gold passage higher")
        if present and not any(item in final_ids for item in expected):
            tags.append("expected candidate omitted from final ranking (diversification/truncation)")
        first = passages.get(final_ids[0]) if final_ids else None
        gold = [passages[item] for item in expected]
        if first and any(first["document_id"] == item["document_id"] for item in gold):
            tags.append("winner and gold are different chunks in the same document")
        if first and any(first["text"].strip() == item["text"].strip() for item in gold):
            tags.append("identical evidence in another chunk; gold label review needed")
        if first and any(first["document_id"] == item["document_id"] + "-related" for item in gold):
            tags.append("curated near-topic distractor won")
        after = record.get("cross_encoder_candidates", [])
        if after and first:
            scores = {item["chunk_id"]: item["score"] for item in after}
            if any(scores.get(item) is not None and scores.get(item) == scores.get(final_ids[0]) for item in expected):
                tags.append("gold and winner have equal CrossEncoder scores")
        if alternative_won:
            tags = ["curated relevant alternative outranks canonical gold"]
        patterns.update(tags)
        failures.append(dict(case_id=record["case_id"], question=record["question"],
                             classification="canonical gold not #1; curated relevant alternative is #1" if alternative_won else "no curated relevant passage at #1",
                             primary_gold_chunk_id=primary, relevant_chunk_ids=expected, relevant_top1=alternative_won,
                             expected=[dict(chunk_id=item["id"], document_id=item["document_id"], page=item["page"],
                                            final_rank=final_ids.index(item["id"])+1 if item["id"] in final_ids else None,
                                            initial_rank=initial_ids.index(item["id"])+1 if item["id"] in initial_ids else None,
                                            preview=" ".join(item["text"].split())[:240]) for item in gold],
                             winner=dict(chunk_id=final_ids[0], document_id=first["document_id"] if first else None,
                                         preview=" ".join(first["text"].split())[:240] if first else None) if final_ids else None,
                             expected_in_candidates=present, initial_ranking=record.get("initial_ranking", []),
                             cross_encoder_ranking=after, final_ranking=record.get("final_ranking", []), patterns=tags))
    patterns = Counter(tag for item in failures if not item["relevant_top1"] for tag in item["patterns"])
    return dict(failures=failures, patterns=dict(patterns), relevant_alternative_wins=sum(item["relevant_top1"] for item in failures),
                true_top1_failures=sum(not item["relevant_top1"] for item in failures),
                note="Groups overlap. Ranking data cannot establish semantic equivalence, claim support, or gold-label ambiguity unless passages are identical. Review the previews manually.")


def print_failures(report, strategy):
    print("\nRANKING FAILURES")
    for item in report["failures"]:
        print(f"\n{item['case_id']}: {item['question']}")
        print(f"{item['classification']} | canonical={item['primary_gold_chunk_id']}")
        print(f"Expected in initial candidates: {item['expected_in_candidates']}")
        for gold in item["expected"]:
            print(f"Gold: {gold['chunk_id']} | document={gold['document_id']} | initial rank={gold['initial_rank']} | final rank={gold['final_rank']}\n  {gold['preview']}")
        print(f"#1: {item['winner']}")
        for label, key in (("Dense retrieval" if strategy == "dense" else "Fused RRF retrieval", "initial_ranking"),
                           ("CrossEncoder (before diversification)", "cross_encoder_ranking"), ("Final ranking", "final_ranking")):
            print(label + ":")
            if not item[key]:
                print("  Scores unavailable in this saved result; no inference was rerun.")
            for row in item[key]:
                print(f"  #{row['rank']} {row['chunk_id']} | document={row['document_id']} | source={row['title']} | score={row['score']}")
    print(f"\nTrue top-1 failures: {report['true_top1_failures']} | Relevant alternative wins: {report['relevant_alternative_wins']}")
    for label, count in report["patterns"].items():
        print(f"  {count} · {label}")
    print(report["note"])


def evaluate_cases(data, retrieve, rerank, checkpoint=lambda records: None):
    passages = {item["id"]: item for item in data["passages"]}
    records = []
    for case in data["cases"]:
        clock = time.perf_counter()
        sources = retrieve(case["question"])
        retrieval_ms = (time.perf_counter() - clock) * 1000
        raw_ids = [item.chunk_id for item in sources]
        initial_ranking = ranking_snapshot(sources)
        clock = time.perf_counter()
        ranked = rerank(case["question"], sources)
        reranking_ms = (time.perf_counter() - clock) * 1000
        cross_encoder = ranking_snapshot(sorted(sources, key=lambda item: item.relevance_score if item.relevance_score is not None else float("-inf"), reverse=True))
        records.append(dict(case_id=case["id"], question=case["question"], style=case["style"], cohort=case.get("cohort", "original"),
                            expected_chunk_ids=case["expected_chunk_ids"], expected_document_ids=case["expected_document_ids"],
                            primary_gold_chunk_id=case.get("primary_gold_chunk_id", case["expected_chunk_ids"][0]),
                            relevant_chunk_ids=case.get("relevant_chunk_ids", case["expected_chunk_ids"]),
                            retrieved_ids=raw_ids, ranked_ids=[item.chunk_id for item in ranked],
                            initial_ranking=initial_ranking, cross_encoder_candidates=cross_encoder, final_ranking=ranking_snapshot(ranked),
                            retrieval=ranking_metrics(raw_ids, case.get("relevant_chunk_ids", case["expected_chunk_ids"])),
                            reranked=ranking_metrics([item.chunk_id for item in ranked], case.get("relevant_chunk_ids", case["expected_chunk_ids"])),
                            retrieval_ms=retrieval_ms, reranking_ms=reranking_ms,
                            source_ids_checked=len(sources) + len(ranked),
                            valid_source_ids=sum(source_valid(item, passages) for item in [*sources, *ranked])))
        checkpoint(records)
    return records


def summary(records):
    checked = sum(item["source_ids_checked"] for item in records)
    result = {stage: {name: statistics.mean(item[stage][name] for item in records)
                      for name in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr")}
              for stage in ("retrieval", "reranked")} if records else {}
    result.update(retrieval_p50_ms=statistics.median(item["retrieval_ms"] for item in records) if records else None,
                  reranking_p50_ms=statistics.median(item["reranking_ms"] for item in records) if records else None,
                  source_id_validity=sum(item["valid_source_ids"] for item in records) / checked if checked else None)
    return result


def save(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)


def cleanup_store(directory):
    path = Path(directory).resolve()
    if path.parent != Path(tempfile.gettempdir()).resolve() or not path.name.startswith("atlas-rag-"):
        raise ValueError("Refusing to clean a directory outside the evaluation temporary store")
    for attempt in range(3):
        try:
            shutil.rmtree(path)
            return None
        except FileNotFoundError:
            return None
        except PermissionError as error:
            if attempt == 2:
                return f"Temporary evaluation store could not be removed: {path} ({error}). Remove it after the locking process exits."
            time.sleep(.1)


def worker(dataset_path, output, store_directory, strategy="dense"):
    started = time.perf_counter()
    output = Path(output)
    result = dict(version=1, kind="rag_baseline", status="running", records=[])
    try:
        # Heavy imports and model loading stay out of unit tests and the timeout supervisor.
        import chromadb
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from unittest.mock import patch
        from backend.core.config import EMBEDDING_MODEL, RERANKER_MODEL
        from backend.ingestion.ingest_pdf import chunk_text
        from backend.retrieval import retriever, reranker
        from backend.retrieval.hybrid import LexicalIndex, retrieve_hybrid

        data = load_dataset(dataset_path)
        result.update(dataset_version=data["version"], dataset_sha256=hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
                      relevant_alternatives=sum(len(case.get("relevant_chunk_ids", case["expected_chunk_ids"])) - 1 for case in data["cases"]),
                      metric_semantics="Recall@K is any-curated-relevant hit rate; MRR uses first relevant rank",
                      planned_cases=len(data["cases"]), embedding_model=EMBEDDING_MODEL, reranker_model=RERANKER_MODEL,
                      settings=dict(retrieval_k=5, reranker_k=6, diversify=True),
                      citation_validation="Retained source/chunk IDs, document IDs, pages and exact passages; no generated-answer citation evaluation.")
        save(output, result)
        result["strategy"] = strategy
        if strategy == "hybrid":
            result["settings"].update(bm25_k=5, fused_k=10, rrf_constant=60, bm25_k1=1.5, bm25_b=.75)
        setup = time.perf_counter()
        embedding = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        ranking = CrossEncoder(RERANKER_MODEL, local_files_only=True)
        # The supervisor removes the store after this process exits and Windows releases all handles.
        with nullcontext(store_directory) as directory:
            client = chromadb.PersistentClient(path=directory)
            collection = client.get_or_create_collection("atlas_rag_eval")
            for passage in data["passages"]:
                if chunk_text(passage["text"]) != [passage["text"]]:
                    raise ValueError("Gold page must map to one existing ingestion chunk")
            collection.upsert(ids=[item["id"] for item in data["passages"]],
                              documents=[item["text"] for item in data["passages"]],
                              embeddings=embedding.encode([item["text"] for item in data["passages"]]).tolist(),
                              metadatas=[dict(document_id=item["document_id"], source=item["document_id"] + ".pdf",
                                              page=item["page"], chunk=0) for item in data["passages"]])
            lexical_index = LexicalIndex(collection) if strategy == "hybrid" else None
            result["setup_seconds"] = time.perf_counter() - setup
            def checkpoint(records):
                result.update(records=list(records), summary=summary(records), total_seconds=time.perf_counter() - started)
                result["cohorts"] = {cohort: dict(cases=len(group), summary=summary(group))
                                     for cohort in sorted({item["cohort"] for item in records})
                                     if (group := [item for item in records if item["cohort"] == cohort])}
                save(output, result)
            with patch.object(retriever, "get_document_collection", return_value=collection), \
                 patch.object(retriever, "get_embedding_model", return_value=embedding), \
                 patch.object(reranker, "get_reranker", return_value=ranking):
                retrieval = (lambda query: retrieve_hybrid(query, lexical_index=lexical_index)) if strategy == "hybrid" else retriever.retrieve
                evaluate_cases(data, retrieval, reranker.rerank, checkpoint)
        result["status"] = "completed"
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
    result["total_seconds"] = time.perf_counter() - started
    save(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET, help="Gold JSON dataset (default: rag-v3.json)")
    parser.add_argument("--strategy", choices=("dense", "hybrid"), default="dense")
    parser.add_argument("--failures", action="store_true", help="Print detailed non-gold top-1 rankings")
    parser.add_argument("--results", type=Path, help="Analyze saved JSON without rerunning retrieval")
    parser.add_argument("--timeout", type=float, default=120, help="Hard total timeout in seconds (default: 120)")
    args = parser.parse_args()
    if args.results:
        result = json.loads(args.results.read_text(encoding="utf-8"))
        data = load_dataset(args.dataset)
        digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
        if result.get("dataset_sha256") and result["dataset_sha256"] != digest:
            parser.error("Saved result dataset hash differs; select its original --dataset")
        print_failures(failure_analysis(result["records"], data), result.get("strategy", "dense"))
        return 0
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    directory = ROOT / "data" / "benchmarks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-rag-" + uuid4().hex[:8] + ".json")
    save(path, dict(version=1, kind="rag_baseline", status="running", records=[]))
    started = time.perf_counter()
    store_directory = tempfile.mkdtemp(prefix="atlas-rag-")
    process = multiprocessing.get_context("spawn").Process(target=worker, args=(str(args.dataset.resolve()), str(path), store_directory, args.strategy))
    cleanup_warning = None
    try:
        process.start()
        process.join(args.timeout)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
        if process.is_alive():
            cleanup_warning = f"Worker has not exited; temporary store retained at {store_directory}."
        else:
            cleanup_warning = cleanup_store(store_directory)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["status"] == "running":
        result.update(status="timeout" if time.perf_counter() - started >= args.timeout else "error",
                      error="Stopped before completion; available cases retained.")
    result.update(total_seconds=time.perf_counter() - started, timeout_seconds=args.timeout)
    if args.failures:
        result["failure_analysis"] = failure_analysis(result["records"], load_dataset(args.dataset))
    if cleanup_warning:
        result["cleanup_warning"] = cleanup_warning
    save(path, result)
    metrics = result.get("summary", {})
    print("ATLAS RAG BASELINE\n--------------------------------")
    print(f"Strategy            {args.strategy}")
    print(f"Cases               {len(result['records'])}/{result.get('planned_cases', '?')}")
    print(f"Relevant alternatives: {result.get('relevant_alternatives', 0)}")
    for label, key in (("Recall@1", "recall_at_1"), ("Recall@3", "recall_at_3"), ("Recall@5", "recall_at_5"), ("MRR", "mrr")):
        value = metrics.get("reranked", {}).get(key)
        print(f"{label:20}{value:.3f}" if value is not None else f"{label:20}n/a")
    for label, key in (("Retrieval p50", "retrieval_p50_ms"), ("Reranker p50", "reranking_p50_ms")):
        value = metrics.get(key)
        print(f"{label:20}{value:.1f} ms" if value is not None else f"{label:20}n/a")
    validity = metrics.get("source_id_validity")
    print(f"Source-ID validity  {validity:.1%}" if validity is not None else "Source-ID validity  n/a")
    print(f"Total               {result['total_seconds']:.1f} s\nStatus              {result['status']}\nSaved               {path}")
    if result.get("error"):
        print(result["error"])
        print("Ensure backend dependencies and the configured embedding/reranker models are cached locally. No models are downloaded by this command.")
    if cleanup_warning:
        print(f"Cleanup warning: {cleanup_warning}")
    if args.failures:
        print_failures(result["failure_analysis"], args.strategy)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
