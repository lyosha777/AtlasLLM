import json
import ollama

from backend.core.config import PLANNER_MODEL

def plan(message: str, history: list) -> dict:

    recent_history = history[-6:]

    prompt = f"""
You are the planner for an AI assistant called Atlas.

Decide what Atlas should do with the latest user message.

Available actions:

chat
Use when no external information is needed.
Examples: greetings, thanks, casual conversation.

document_search
Use when information should be retrieved from the user's documents.

Use conversation history to resolve follow-up references such as:
"it", "that", "those", "the transport one", etc.

Conversation history:
{recent_history}

Latest message:
{message}

Return valid JSON only:

{{
    "action": "chat" or "document_search",
    "query": null or "standalone search query"
}}
"""

    response = ollama.chat(
        model=PLANNER_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    try:
        result = json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        return {
            "action": "chat",
            "query": None
        }

    if result.get("action") not in {"chat", "document_search"}:
        return {
            "action": "chat",
            "query": None
        }

    return result