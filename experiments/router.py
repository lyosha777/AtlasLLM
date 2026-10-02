import ollama


def classify_intent(message: str) -> str:

    prompt = f"""
Classify the user's message into exactly ONE category:

CHAT
The user is chatting, thanking, greeting, joking, or asking something
that does not require information from their uploaded documents.

DOCUMENT
The user is asking a factual question that should be answered using
their uploaded knowledge base.

Examples:

"thanks bro" -> CHAT
"hello" -> CHAT
"how are you?" -> CHAT

"What is the OSI model?" -> DOCUMENT
"Explain TCP according to my documents" -> DOCUMENT
"What does the networking material say about routers?" -> DOCUMENT

User message:
{message}

Respond with ONLY:
CHAT
or
DOCUMENT
"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    intent = response["message"]["content"].strip().upper()

    if intent not in {"CHAT", "DOCUMENT"}:
        return "CHAT"

    return intent