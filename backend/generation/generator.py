import ollama
import ollama

from backend.core.config import GENERATION_MODEL

def generate_answer(query: str, chunks: list):

    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[SOURCE {i}]\n"
            f"Document: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Content:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are Atlas, a research assistant.

Answer the question using ONLY the provided sources.

Rules:
- Do not use outside knowledge.
- If the sources do not contain enough information, say so.
- Cite information using [SOURCE 1], [SOURCE 2], etc.
- Do not invent sources or citations.

SOURCES:

{context}

QUESTION:
{query}

ANSWER:
"""

    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]