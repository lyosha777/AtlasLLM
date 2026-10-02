from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# Load an embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


documents = [
    "Dogs are common household pets.",
    "Python is a programming language.",
    "Cats make great pets.",
    "The CPU executes instructions in a computer.",
    "Paris is the capital of France."
]

query = "animals people keep at home"


# Convert text into vectors
document_embeddings = model.encode(documents)
query_embedding = model.encode(query)


# Compare the query against every document
scores = cos_sim(query_embedding, document_embeddings)[0]


# Sort from most similar to least similar
results = sorted(
    zip(documents, scores),
    key=lambda x: x[1],
    reverse=True
)


print(f'\nQuery: "{query}"\n')

for document, score in results:
    print(f"{score:.3f}  |  {document}")