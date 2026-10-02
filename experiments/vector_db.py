import chromadb
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

# Creates/opens a persistent database on your computer
client = chromadb.PersistentClient(path="./data/chroma")

# A collection is roughly like a table for our vectors
collection = client.get_or_create_collection(
    name="atlas_knowledge"
)


documents = [
    "Dogs are common household pets.",
    "Cats make great pets and often live indoors.",
    "Python is a popular programming language used in AI and data science.",
    "The CPU executes instructions and performs calculations in a computer.",
    "RAM temporarily stores data that programs are actively using.",
    "TCP provides reliable and ordered delivery of data between applications.",
    "UDP sends datagrams without guaranteeing delivery or ordering.",
    "Machine learning allows computers to learn patterns from data.",
    "Large language models generate text by predicting tokens.",
]


# ---------- INGESTION ----------

embeddings = model.encode(documents).tolist()

collection.upsert(
    ids=[f"doc_{i}" for i in range(len(documents))],
    documents=documents,
    embeddings=embeddings,
)


print(f"Atlas contains {collection.count()} documents.\n")


# ---------- RETRIEVAL ----------

while True:
    query = input("Search > ")

    if query.lower() == "quit":
        break

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print("\nTop results:")

    for document, distance in zip(
        results["documents"][0],
        results["distances"][0]
    ):
        print(f"{distance:.3f} | {document}")

    print()