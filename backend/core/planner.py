import json
import ollama

from backend.core.config import PLANNER_MODEL
from backend.models.ollama_provider import InferenceError


def plan(message: str, history: list, has_attachments: bool = False) -> dict:
    recent_history = history[-6:]

    prompt = f"""
You are the tool planner for Atlas, an AI research assistant.

Your job is to decide whether Atlas should answer using its internal
knowledge, search the user's documents, or search the web.

DOCUMENTS

Use documents when:
- the user refers to their files, PDFs, notes, documents, or uploads
- the answer specifically depends on information in their knowledge base

WEB

Use the web when external factual verification would significantly
improve reliability.

Use the web for:
- current or changing information
- news, weather, prices, releases, schedules, current events
- identifying songs, quotes, lyrics, books, movies, people, products, etc.
- specific facts about named people, companies, products, places, or events
- specific dates, statistics, versions, releases, or factual claims
- obscure facts the model may not reliably know
- requests where the user explicitly asks to search, verify, check, or look up something

Do NOT use the web just because a question is factual.

Do not use the web for:
- casual conversation
- writing or rewriting
- summarization of text already provided
- simple calculations
- reasoning problems
- programming help that does not require current information
- explanations of stable, well-known concepts
- general educational questions the model can reliably explain

Examples:

"hello bro"
-> documents=false, web=false, fresh=false

"Explain TCP"
-> documents=false, web=false, fresh=false

"What's the difference between RAM and storage?"
-> documents=false, web=false, fresh=false

"Write me an email to my professor"
-> documents=false, web=false, fresh=false

"According to my networking PDF, explain TCP"
-> documents=true, web=false, fresh=false

"Who invented the telephone?"
-> documents=false, web=true, fresh=false

"What song starts with 'somebody once told me'?"
-> documents=false, web=true, fresh=false

"Who said 'I think therefore I am'?"
-> documents=false, web=true, fresh=false

"When was the RTX 4090 released?"
-> documents=false, web=true, fresh=false

"What is the latest NVIDIA GPU news?"
-> documents=false, web=true, fresh=true

"What happened in Armenia today?"
-> documents=false, web=true, fresh=true

"What is the current price of Bitcoin?"
-> documents=false, web=true, fresh=true

FRESHNESS

fresh=true only when the answer depends on information that may have
changed recently.

Examples include:
- latest
- current
- today
- yesterday
- this week
- recently
- breaking news
- current prices
- current versions
- current availability

A web search does NOT automatically mean fresh=true.

SEARCH QUERY

When using documents or web, produce a concise standalone search query.

Resolve references using conversation history.

For identification requests, turn the user's wording into a useful
search query.

Example:

User:
"somebody once told me..."

Context indicates they are trying to identify a song.

search_query:
"song lyrics somebody once told me"

Do not answer the user's question yourself.
Only make the routing decision.

Conversation history:
{recent_history}

Latest message:
{message}

PDF files attached to this message: {has_attachments}
Attached files are an explicit request to use documents. You may also
select web search when it would help answer the request.

Return ONLY valid JSON:

{{
    "use_documents": true or false,
    "use_web": true or false,
    "fresh": true or false,
    "search_query": "standalone search query" or null
}}
"""

    try:
        response = ollama.chat(
            model=PLANNER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
    except Exception:
        raise InferenceError("Ollama planning failed. Check that Ollama and the configured planner model are available.") from None

    try:
        result = json.loads(response["message"]["content"])

        use_documents = bool(
            result.get("use_documents", False)
        )

        use_web = bool(
            result.get("use_web", False)
        )

        fresh = bool(
            result.get("fresh", False)
        )

        search_query = result.get("search_query")

        if has_attachments:
            use_documents = True

        if fresh:
            use_web = True

        if not use_documents and not use_web:
            search_query = None
            fresh = False

        return {
            "use_documents": use_documents,
            "use_web": use_web,
            "fresh": fresh,
            "search_query": search_query
        }

    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        return {
            "use_documents": has_attachments,
            "use_web": False,
            "fresh": False,
            "search_query": None
        }
