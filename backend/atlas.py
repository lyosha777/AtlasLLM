import time

from backend.core.planner import plan
from backend.retrieval.retriever import retrieve
from backend.retrieval.web_retriever import retrieve_web
from backend.retrieval.reranker import rerank
from backend.generation.generator import generate_answer
from backend.generation.chat import chat


history = []

print("\n============================")
print("           ATLAS")
print("============================")
print("Chat normally or ask Atlas to research something.")
print("Type 'quit' to exit.\n")


while True:
    query = input("You > ").strip()

    if query.lower() == "quit":
        break

    if not query:
        continue

    total_start = time.perf_counter()

    start = time.perf_counter()
    decision = plan(query, history)
    planner_time = time.perf_counter() - start

    use_documents = decision["use_documents"]
    use_web = decision["use_web"]
    fresh = decision["fresh"]
    search_query = decision["search_query"] or query

    print(f"\n[Planner: {planner_time:.2f}s]")
    print(
        f"[Documents: {'yes' if use_documents else 'no'} | "
        f"Web: {'yes' if use_web else 'no'} | "
        f"Fresh: {'yes' if fresh else 'no'}]"
    )

    if not use_documents and not use_web:
        start = time.perf_counter()
        answer = chat(query, history)
        generation_time = time.perf_counter() - start

    else:
        sources = []

        if use_documents:
            start = time.perf_counter()
            document_sources = retrieve(search_query)
            document_time = time.perf_counter() - start

            print(f"[Document retrieval: {document_time:.2f}s]")
            sources.extend(document_sources)

        if use_web:
            start = time.perf_counter()

            web_sources = retrieve_web(
                search_query,
                fresh=fresh
            )

            web_time = time.perf_counter() - start

            print(f"[Web retrieval: {web_time:.2f}s]")
            sources.extend(web_sources)

        print(f"[Found {len(sources)} candidates]")

        start = time.perf_counter()
        sources = rerank(search_query, sources, top_k=6)
        reranker_time = time.perf_counter() - start

        print(f"[Reranking: {reranker_time:.2f}s]")
        print(f"[Selected {len(sources)} sources]")

        if sources:
            print("\nSources:")

            for i, source in enumerate(sources, start=1):
                if source.source_type == "document":
                    print(f"[{i}] {source.title} (page {source.page})")

                elif source.source_type == "web":
                    print(f"[{i}] {source.title}")
                    print(f"    {source.url}")

        start = time.perf_counter()
        answer = generate_answer(query, sources)
        generation_time = time.perf_counter() - start

    total_time = time.perf_counter() - total_start

    print(f"\n[Generation: {generation_time:.2f}s]")
    print(f"[Total: {total_time:.2f}s]\n")

    history.append({
        "role": "user",
        "content": query
    })

    history.append({
        "role": "assistant",
        "content": answer
    })