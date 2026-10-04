from concurrent.futures import ThreadPoolExecutor, as_completed

import trafilatura


def read_webpage(url: str) -> dict | None:
    """Download a webpage and extract its useful main text."""

    try:
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return None

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )

        if not text or len(text.strip()) < 200:
            return None

        return {
            "url": url,
            "text": text.strip(),
        }

    except Exception:
        return None


def read_webpages(urls: list[str], max_workers: int = 5) -> list[dict]:
    """Read several webpages concurrently."""

    pages = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(read_webpage, url): url
            for url in urls
        }

        for future in as_completed(futures):
            result = future.result()

            if result:
                pages.append(result)

    return pages