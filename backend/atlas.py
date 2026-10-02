from backend.core.planner import plan
from backend.retrieval.retriever import retrieve
from backend.generation.generator import generate_answer
from backend.generation.chat import chat


history = []


print("\n============================")
print("           ATLAS")
print("============================")
print("Chat normally or ask about your documents.")
print("Type 'quit' to exit.\n")


while True:
    query = input("You > ").strip()

    if query.lower() == "quit":
        break

    if not query:
        continue

    # One LLM call handles both routing and query rewriting
    decision = plan(query, history)

    action = decision["action"]
    search_query = decision["query"]

    print(f"\n[Action -> {action}]")

    if search_query:
        print(f"[Query -> {search_query}]")

    # Normal conversation
    if action == "chat":
        answer = chat(query, history)

    # Document RAG
    elif action == "document_search":
        chunks = retrieve(search_query)

        print("\nRetrieved:")

        for i, chunk in enumerate(chunks, start=1):
            print(
                f"[SOURCE {i}] "
                f"{chunk['source']} "
                f"(page {chunk['page']}) "
                f"distance={chunk['distance']:.3f}"
            )

        answer = generate_answer(query, chunks)

    else:
        answer = "I couldn't determine how to handle that request."

    print(f"\nAtlas > {answer}\n")

    history.append({
        "role": "user",
        "content": query
    })

    history.append({
        "role": "assistant",
        "content": answer
    })