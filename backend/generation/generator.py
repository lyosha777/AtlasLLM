from collections.abc import Iterator

from backend.core.config import GENERATION_MODEL
from backend.core.source import Source
from backend.models.ollama_provider import provider


def build_context(sources: list[Source]) -> str:
    parts = []

    for index, source in enumerate(sources, start=1):
        metadata = [
            f"Title: {source.title}",
            f"Type: {source.source_type}"
        ]

        if source.page is not None:
            metadata.append(f"Page: {source.page}")

        if source.url:
            metadata.append(f"URL: {source.url}")

        parts.append(
            f"[SOURCE {index}]\n"
            + "\n".join(metadata)
            + f"\nContent:\n{source.text}"
        )

    return "\n\n".join(parts)


def answer_events(
    question: str,
    sources: list[Source],
    inference: dict | None = None,
) -> Iterator[dict]:

    if not sources:
        yield {"type": "token", "content": "I couldn't find enough reliable information to answer that."}
        return

    context = build_context(sources)

    prompt = f"""
You are Atlas, a grounded research assistant.

Answer the user's question using only factual information supported by
the provided sources.

Rules:
- Answer directly.
- Be concise by default.
- Give more detail when the user explicitly asks for it.
- Do not repeat information.
- Treat the supplied sources as authoritative for this answer.
- Do not contradict the sources using pretrained knowledge.
- Do not speculate beyond the supplied information.
- Cite factual claims using [SOURCE 1], [SOURCE 2], etc.
- If sources disagree, briefly explain the disagreement.
- If the sources are insufficient, say so.
- Do not add unnecessary disclaimers or conclusions.

Question:
{question}

Sources:
{context}
"""

    yield from provider.stream(
        [
            {
                "role": "user",
                "content": prompt
            }
        ], inference or provider.resolve(GENERATION_MODEL), research=True,
    )


def generate_answer_stream(question: str, sources: list[Source], inference: dict | None = None) -> Iterator[str]:
    for event in answer_events(question, sources, inference):
        if event["type"] == "token":
            yield event["content"]


def generate_answer(
    question: str,
    sources: list[Source]
) -> str:

    return "".join(
        generate_answer_stream(question, sources)
    )
