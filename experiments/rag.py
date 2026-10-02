import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# -------------------------
# SETUP
# -------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./data/chroma")

collection = client.get_collection(
    name="atlas_knowledge"
)


# -------------------------
# RETRIEVAL
# -------------------------

def retrieve(query, n_results=3):

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results["documents"][0]


# -------------------------
# GENERATION
# -------------------------

def generate_answer(query, documents):

    context = "\n".join(documents)

    prompt = f"""
You are Atlas, a knowledge assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information to answer,
say: "I don't have enough information to answer that."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


# -------------------------
# RAG LOOP
# -------------------------

print("\nAtlas RAG")
print("Type 'quit' to exit.\n")


while True:

    query = input("You > ")

    if query.lower() == "quit":
        break

    documents = retrieve(query)

    print("\nRetrieved:")
    for document in documents:
        print(f"- {document}")

    answer = generate_answer(query, documents)

    print(f"\nAtlas > {answer}\n")