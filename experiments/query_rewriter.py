import ollama


def rewrite_query(message: str, history: list) -> str:

    if not history:
        return message

    recent_history = history[-6:]

    prompt = f"""
Rewrite the user's latest message as a standalone search query.

Use the conversation history only to resolve references such as
"it", "that", "they", "those", or follow-up questions.

Do NOT answer the question.
Do NOT add information the user didn't ask for.
If the message is already standalone, return it unchanged.

Conversation history:
{recent_history}

Latest message:
{message}

Return ONLY the rewritten query.
"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"].strip()