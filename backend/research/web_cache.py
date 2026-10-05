from urllib.parse import urlsplit, urlunsplit

from backend.research.schemas import MAX_URLS
from backend.tools.web_reader import read_webpages


def canonical_url(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))


class WebCache:
    def __init__(self):
        self.pages = {}
        self.urls_considered = set()
        self.last_operation = {}
        self.chunks_considered = 0

    def read(self, urls):
        urls = list(dict.fromkeys(canonical_url(url) for url in urls))
        self.urls_considered.update(urls)
        remaining = MAX_URLS - len(self.pages)
        pending = [url for url in urls if url not in self.pages][:remaining]
        # Remember unsuccessful fetches too; a failed URL is never repeatedly downloaded.
        for url in pending:
            self.pages[url] = None
        pages = read_webpages(pending) if pending else []
        for page in pages:
            self.pages[canonical_url(page["url"])] = page
        self.last_operation = dict(urls=urls, fetched=pending,
                                   failed=[url for url in pending if self.pages[url] is None],
                                   budget_skipped=[url for url in urls if url not in self.pages])
        return [self.pages[url] for url in urls if self.pages.get(url)]

    def metrics(self):
        return dict(urls_considered=len(self.urls_considered), urls_fetched=len(self.pages),
                    pages_fetched=sum(page is not None for page in self.pages.values()),
                    fetch_failures=sum(page is None for page in self.pages.values()),
                    web_chunks_considered=self.chunks_considered)
