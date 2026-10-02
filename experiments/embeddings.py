from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


documents = [
    "Dogs are common household pets.",
    "Cats make great pets and often live indoors.",
    "Python is a popular programming language used in AI and data science.",
    "JavaScript is commonly used to build interactive websites.",
    "The CPU executes instructions and performs calculations in a computer.",
    "RAM temporarily stores data that programs are actively using.",
    "Paris is the capital city of France.",
    "Brussels is the capital city of Belgium.",
    "TCP provides reliable and ordered delivery of data between applications.",
    "UDP sends datagrams without guaranteeing delivery or ordering.",
    "Machine learning allows computers to learn patterns from data.",
    "Large language models generate text by predicting tokens.",
]


# We only need to embed the documents ONCE.
document_embeddings = model.encode(documents)


print("\nAtlas Semantic Search")
print("Type 'quit' to exit.\n")


while True:

    query = input("Search > ")

    if query.lower() == "quit":
        break

    # Convert the user's query into the same vector space
    query_embedding = model.encode(query)

    # Compare query against all documents
    scores = cos_sim(query_embedding, document_embeddings)[0]

    # Rank documents by similarity
    results = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop results:")

    for document, score in results[:3]:
        print(f"{score:.3f} | {document}")

    print()