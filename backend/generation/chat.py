import ollama
from backend.core.config import CHAT_MODEL

def chat(message: str, history: list):

    messages = [
        {
            "role": "system",
            "content": (
                "You are Atlas, a helpful AI assistant. "
                "Respond naturally and concisely."
            )
        }
    ]

    messages.extend(history)

    messages.append({
        "role": "user",
        "content": message
    })

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=messages
    )

    return response["message"]["content"]