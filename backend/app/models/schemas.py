"""Pydantic schemas for repository ingestion and its metadata report."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class FileKind(str, Enum):
    """How a file participates in the analysis."""

    SOURCE = "source"
    CONFIG = "config"
    DOCUMENTATION = "documentation"
    DATA = "data"
    BUILD = "build"
    BINARY = "binary"
    OTHER = "other"


class RepositoryRequest(BaseModel):
    """Payload for POST /api/v1/repos."""

    url: str = Field(min_length=1, max_length=2048)
    ref: Optional[str] = Field(
        default=None,
        description="Reserved. v0.1 always clones the repository's default branch.",
    )


class FileEntry(BaseModel):
    """One discovered file, with enough provenance for later evidence retrieval.

    ``path`` is relative to the repository root and is the canonical reference
    future milestones (indexing, Q&A, judging) should attach citations to.
    """

    path: str
    language: Optional[str] = None
    category: FileKind
    size_bytes: int


class LanguageStats(BaseModel):
    language: str
    file_count: int
    size_bytes: int


class CategoryStats(BaseModel):
    category: FileKind
    file_count: int
    size_bytes: int


class FrameworkInfo(BaseModel):
    name: str
    kind: str  # "framework" | "library" | "tooling" | "package_manager"
    source: str  # relative path providing the evidence


class IgnoredDirectory(BaseModel):
    path: str
    reason: str


class DirectoryNode(BaseModel):
    name: str
    path: str
    file_count: int
    size_bytes: int
    truncated: bool = False
    children: list["DirectoryNode"] = Field(default_factory=list)


class RepositoryManifest(BaseModel):
    """Structured metadata report for one ingested GitHub repository.

    ``languages`` / ``frameworks`` are derived from *deterministic* file
    signals only. ``readme_claims`` are keyword mentions scraped from the
    README and are explicitly *unverified claims*, not detected facts.
    """

    schema_version: int
    ingested_at: datetime
    id: str
    owner: str
    repo: str
    github_url: str
    clone_ref: Optional[str] = None
    commit_hash: Optional[str] = None
    default_branch: Optional[str] = None
    readme_present: bool = False
    readme_claims: list[str] = Field(default_factory=list)
    total_files: int = 0
    source_files: int = 0
    total_size_bytes: int = 0
    languages: list[LanguageStats] = Field(default_factory=list)
    categories: list[CategoryStats] = Field(default_factory=list)
    frameworks: list[FrameworkInfo] = Field(default_factory=list)
    package_manager: Optional[str] = None
    config_files: list[str] = Field(default_factory=list)
    ignored_directories: list[IgnoredDirectory] = Field(default_factory=list)
    directory_structure: Optional[DirectoryNode] = None
    file_inventory: list[FileEntry] = Field(default_factory=list)
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """One evidence chunk returned by a search, with exact provenance.

    ``file_path`` is relative to the repository root and ``start_line``/
    ``end_line`` are 1-indexed, inclusive — together they are the citation
    anchor for future LLM answers (M3+).
    """

    file_path: str
    start_line: int
    end_line: int
    language: Optional[str] = None
    score: float = 0.0
    content: str


class SearchResponse(BaseModel):
    """Structured results for ``GET /repos/{owner}/{repo}/search``."""

    query: str
    repo_id: str
    total: int
    results: list[SearchResult] = Field(default_factory=list)


class IndexSummary(BaseModel):
    """Result of an (re-)indexing operation."""

    repo_id: str
    files_indexed: int
    chunks_created: int


# ---------------------------------------------------------------------------
# M3 — evidence-grounded Q&A
# ---------------------------------------------------------------------------


class QuestionRequest(BaseModel):
    """Payload for POST /repos/{owner}/{repo}/ask."""

    question: str = Field(min_length=1, max_length=2000)
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="How many evidence chunks to retrieve (default: QA_TOP_K env).",
    )


class Citation(BaseModel):
    """One citation attached to an answer, resolved to a real evidence chunk.

    ``id`` is the evidence label used inside the answer text ("E1"), and
    ``file_path``/``start_line``/``end_line`` are the exact 1-indexed,
    inclusive provenance of the chunk the model's claim rests on.
    """

    id: str
    file_path: str
    start_line: int
    end_line: int
    language: Optional[str] = None
    content: Optional[str] = None


class AnswerResponse(BaseModel):
    """Structured result of evidence-grounded Q&A.

    ``answer``, ``confidence`` and ``evidence_sufficient`` are what the model
    reported. Validation guarantees every whole citation resolves to a real
    evidence chunk whose ``[E#]`` marker also appears in the answer text — but
    it does NOT verify that a cited chunk actually supports each claim
    (claim-level verification is deferred to M4).

    ``confidence`` is **model-reported** unless the deterministic empty-evidence
    path was taken. It is NOT proof that the evidence supports the answer;
    ``confidence_source`` says exactly which of the two it is.

    ``evidence_grounding`` is derived deterministically from the surviving
    validated citations alone — never from the model's own judgment:
      "none"  → no evidence block is anchored in the answer (includes the
                empty-retrieval short-circuit)
      "cited" → one or more evidence blocks are anchored in the answer
    ("partial" is reserved for M4 claim-level coverage and is never emitted.)
    """

    question: str
    repo_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: str  # "high" | "medium" | "low"
    evidence_sufficient: bool

    # M3.1 — honesty fields: distinguish what the LLM claims from what Northern
    # Star can deterministically verify about the supplied evidence.
    confidence_source: Literal["model", "deterministic"]
    evidence_grounding: Literal["none", "cited"]


DirectoryNode.model_rebuild()