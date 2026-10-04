from collections.abc import Iterator

from backend.core.config import CHAT_MODEL
from backend.models.ollama_provider import provider


def chat_events(
    message: str,
    history: list[dict],
    inference: dict | None = None,
) -> Iterator[dict]:

    messages = [
        {
            "role": "system",
            "content": (
                "You are Atlas, a capable personal research assistant. "
                "Answer directly and naturally. "
                "Be concise by default, but give detail when the user asks for it. "
                "Do not repeat yourself or add unnecessary conclusions."
            )
        }
    ]

    messages.extend(history)

    messages.append({
        "role": "user",
        "content": message
    })

    yield from provider.stream(messages, inference or provider.resolve(CHAT_MODEL))


def chat_stream(message: str, history: list[dict], inference: dict | None = None) -> Iterator[str]:
    for event in chat_events(message, history, inference):
        if event["type"] == "token":
            yield event["content"]


def chat(
    message: str,
    history: list[dict]
) -> str:

    return "".join(
        chat_stream(message, history)
    )
