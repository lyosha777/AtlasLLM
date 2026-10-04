from backend.tools.web_search import search_web


query = input("Search the web > ")

results = search_web(query)

for i, result in enumerate(results, start=1):
    print(f"\n--- RESULT {i} ---")
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Snippet: {result['snippet']}")