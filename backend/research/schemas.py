from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal

MAX_SUBQUESTIONS = 5
MAX_QUERIES = 2
MAX_SEARCHES = 22
MAX_URLS = 18
MAX_PAGE_CHUNKS = 60
MAX_CANDIDATES = 120
CANDIDATES_PER_SEARCH = 5
MAX_EVIDENCE = 12
MAX_CLAIMS = 6
MAX_CLAIM_LINKS = 4
MAX_FOLLOWUPS = 2
PLAN_ATTEMPTS = 2
STRUCTURED_TOKENS = 4096
SYNTHESIS_TOKENS = 2048
CONTEXT_TOKENS = 16384


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Subquestion(Artifact):
    id: str = Field(pattern=r"^sq[1-5]$")
    question: str = Field(min_length=1, max_length=400)
    reason: str = Field(min_length=1, max_length=300)
    search_queries: list[str] = Field(min_length=1, max_length=MAX_QUERIES)

    @model_validator(mode="after")
    def queries_valid(self):
        if any(not query.strip() or len(query) > 400 for query in self.search_queries):
            raise ValueError("Invalid search query")
        return self


class ResearchPlan(Artifact):
    objective: str = Field(min_length=1, max_length=600)
    depth: Literal["quick", "standard", "deep"]
    depth_reason: str = Field(min_length=1, max_length=400)
    subquestions: list[Subquestion] = Field(min_length=1, max_length=MAX_SUBQUESTIONS)

    @model_validator(mode="after")
    def unique_ids(self):
        if len(self.subquestions) > {"quick": 2, "standard": 3, "deep": MAX_SUBQUESTIONS}[self.depth]:
            raise ValueError("Too many subquestions for selected research depth")
        if len({item.id for item in self.subquestions}) != len(self.subquestions):
            raise ValueError("Subquestion IDs must be unique")
        return self


class Claim(Artifact):
    id: str = Field(pattern=r"^c[0-9]{1,2}$")
    statement: str = Field(min_length=1, max_length=400)
    supporting_evidence_ids: list[str] = Field(max_length=MAX_CLAIM_LINKS)
    conflicting_evidence_ids: list[str] = Field(max_length=MAX_CLAIM_LINKS)


class Coverage(Artifact):
    subquestion_id: str
    status: str = Field(pattern=r"^(sufficient|weak|missing)$")
    explanation: str = Field(min_length=1, max_length=400)
    refined_query: str | None = Field(default=None, max_length=400)


class EvidenceAnalysis(Artifact):
    claims: list[Claim] = Field(max_length=MAX_CLAIMS)
    coverage: list[Coverage] = Field(max_length=MAX_SUBQUESTIONS)


def budgets(depth="deep"):
    result = dict(subquestions=MAX_SUBQUESTIONS, queries_per_subquestion=MAX_QUERIES,
                searches=MAX_SEARCHES, urls=MAX_URLS, chunks_per_page=MAX_PAGE_CHUNKS,
                candidates=MAX_CANDIDATES, selected_evidence=MAX_EVIDENCE,
                candidates_per_search=CANDIDATES_PER_SEARCH,
                claims=MAX_CLAIMS, links_per_claim_relation=MAX_CLAIM_LINKS,
                followup_passes=1, followup_searches=MAX_FOLLOWUPS,
                planning_attempts=PLAN_ATTEMPTS, structured_tokens=STRUCTURED_TOKENS,
                synthesis_tokens=SYNTHESIS_TOKENS, context_tokens=CONTEXT_TOKENS)
    if depth == "quick":
        result.update(subquestions=2, searches=3, candidates=15, selected_evidence=5, followup_searches=0)
    elif depth == "standard":
        result.update(subquestions=3, searches=6, candidates=30, selected_evidence=8, followup_searches=1)
    return result
