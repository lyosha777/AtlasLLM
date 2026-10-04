from ddgs import DDGS


def search_web(
    query: str,
    max_results: int = 5,
    fresh: bool = False
) -> list[dict]:

    results = []

    timelimit = "m" if fresh else None

    with DDGS() as ddgs:
        search_results = ddgs.text(
            query,
            max_results=max_results,
            timelimit=timelimit
        )

        for result in search_results:
            url = result.get("href", "")

            if not url:
                continue

            results.append({
                "title": result.get("title", ""),
                "url": url,
                "snippet": result.get("body", "")
            })

    return results