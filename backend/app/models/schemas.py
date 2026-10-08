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


class Claim(BaseModel):
    """A structured claim extracted from a repository's documentation (e.g. README).

    Claims are unverified statements that M4 will evaluate against the evidence
    index. Every claim carries exact file:line provenance from the manifest so
    that verification can attach deterministic citations.

    Verdicts are determined by the verification service (LLM over retrieved
    evidence + deterministic citation validation), never by BM25 presence alone.
    """

    id: str  # e.g. "claim_1" or a slugified version of the claim text
    text: str  # The claim statement as written
    source: str  # Relative path to the file that contains the claim (e.g. "README.md")
    kind: FileKind  # SOURCE, DOCUMENTATION, CONFIG, etc.
    category: str  # Free-text category (e.g. "ai-capability", "performance")
    verdict: Literal["supported", "partially_supported", "unclear", "contradicted"]
    verdict_explanation: str  # Human-readable why, with file:line citations
    evidence_ids: list[str]  # Evidence chunk IDs that influenced the verdict
    repo_id: str  # "owner/repo" for isolation


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

    # ------------------------------------------------------------------
    # M4 — claim-level verdicts (populated when /verify is called).
    # These fields are optional; they exist only on the verification response.
    # ------------------------------------------------------------------
    claims: list["Claim"] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# M5 — Evidence-Based Project Judging
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """Score for one judging dimension."""

    name: str  # e.g., "technical_implementation", "architecture", "claim_integrity", "completeness", "overall_quality"
    score: float  # 0-10
    explanation: str  # Why this score, with evidence citations
    evidence_ids: list[str]  # IDs of evidence blocks that support this judgment


class JudgeResult(BaseModel):
    """Structured result of evidence-based project judging.

    All scores are 0-10 per dimension, overall is 0-100 (deterministic sum).
    Every substantive judgment cites evidence by ID.
    """

    repo_id: str
    overall_score: int  # 0-100, deterministic from dimension scores
    dimensions: list[DimensionScore]
    strengths: list[str]  # Key strengths with evidence IDs
    weaknesses: list[str]  # Key weaknesses with evidence IDs
    recommendations: list[str]  # Actionable recommendations
    claim_integrity_summary: dict[str, int]  # counts: supported, partially_supported, unclear, contradicted
    total_claims: int
    evidence_citations: list[Citation]  # All evidence cited across dimensions


# ---------------------------------------------------------------------------
# M6 — Red-Team / Challenge Engine
# ---------------------------------------------------------------------------


class ChallengeCategory(str, Enum):
    """Category of a challenge."""

    UNSUPPORTED_CLAIM = "unsupported_claim"
    CONTRADICTION = "contradiction"
    MISSING_IMPLEMENTATION = "missing_implementation"
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"
    TESTING = "testing"
    COMPLETENESS = "completeness"
    DOCUMENTATION = "documentation"


class ChallengeSeverity(str, Enum):
    """Severity of a challenge."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Challenge(BaseModel):
    """A red-team challenge against a repository claim or implementation.

    Every challenge is grounded in evidence and relates to an actual
    repository claim or implementation observation.
    """

    id: str  # e.g., "challenge_1"
    claim: str  # The claim or observation being challenged
    challenge: str  # The challenge question/statement
    severity: ChallengeSeverity
    category: ChallengeCategory
    explanation: str  # Human-readable explanation with [E#] citations
    evidence_ids: list[str]  # IDs of evidence blocks supporting the challenge
    repo_id: str  # "owner/repo" for isolation
    confidence: Literal["high", "medium", "low"]  # Confidence in the challenge


class ChallengeResult(BaseModel):
    """Structured result of red-team challenge generation."""

    repo_id: str
    challenges: list[Challenge]
    total_challenges: int
    high_severity: int
    medium_severity: int
    low_severity: int
    evidence_citations: list[Citation]  # All evidence cited across challenges


# ---------------------------------------------------------------------------
# M7 — Evidence-Based Improvement Engine
# ---------------------------------------------------------------------------


class ImprovementPriority(str, Enum):
    """Priority level for an improvement recommendation."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Improvement(BaseModel):
    """A concrete, evidence-backed improvement recommendation.

    Derived from M6 challenges, M5 judge results, and M4 claim verification.
    Every improvement is tied to repository evidence and is technically
    plausible for this specific repository.
    """

    id: str  # e.g., "improvement_1"
    title: str  # Short actionable title
    problem: str  # What is wrong, with [E#] citations
    recommendation: str  # Specific actionable steps, with [E#] citations
    priority: ImprovementPriority
    category: ChallengeCategory  # Reuse M6 categories
    rationale: str  # Why this improvement matters, with evidence
    evidence_ids: list[str]  # IDs of evidence blocks supporting this improvement
    related_challenge_ids: list[str]  # M6 challenge IDs this addresses
    affected_files: list[str]  # Existing repo files that would be modified
    confidence: Literal["high", "medium", "low"]  # Confidence in the recommendation


class ImprovementResult(BaseModel):
    """Structured result of evidence-based improvement generation."""

    repo_id: str
    improvements: list[Improvement]
    total_improvements: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    evidence_citations: list[Citation]  # All evidence cited across improvements


# ---------------------------------------------------------------------------
# M7.1 — Evidence-Grounded Repository Architecture Graph
# ---------------------------------------------------------------------------


class ArchitectureNode(BaseModel):
    """One component in the repository architecture graph.

    Nodes are derived deterministically from the repository's file inventory
    and directory structure — never invented by an LLM. ``files`` lists real
    repository-relative paths; ``evidence_ids`` reference indexed chunks.
    Stable ``id`` values (``project``, ``dir:<path>``, ``file:<path>``) let
    the frontend overlay M5/M6/M7 data later.
    """

    id: str
    label: str
    type: str  # project|directory|module|api|service|model|database|test|config|frontend|backend|utility|unknown
    files: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    description: Optional[str] = None


class ArchitectureEdge(BaseModel):
    """One relationship between two architecture nodes.

    Only emitted when the relationship is established from repository
    evidence (directory containment or a resolved in-repo import).
    """

    source: str
    target: str
    relationship: str  # contains|imports|depends_on|calls|tests|configures|reads_from|writes_to
    evidence_ids: list[str] = Field(default_factory=list)


class ArchitectureResult(BaseModel):
    """Deterministic architecture overview of a repository."""

    repo_id: str
    root: str  # id of the project root node (always "project")
    nodes: list[ArchitectureNode] = Field(default_factory=list)
    edges: list[ArchitectureEdge] = Field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0
    evidence_citations: list[Citation] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# M8.1 — GitHub Repository Discovery (search + trending metadata only;
# never repository analysis evidence)
# ---------------------------------------------------------------------------


class GitHubRepository(BaseModel):
    """Normalized public metadata for one GitHub repository.

    A curated subset of the GitHub REST API repository object — the raw
    response is never exposed. Nullable fields reflect GitHub omitting or
    nulling values. Contains enough (``full_name`` / ``html_url``) for the
    frontend to offer "Analyze with Northern Star" via the M1 endpoint.
    """

    id: int
    full_name: str  # "owner/repo"
    name: str
    owner: str  # login
    html_url: str
    description: Optional[str] = None
    language: Optional[str] = None
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    watchers: int = 0
    topics: list[str] = Field(default_factory=list)
    default_branch: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    pushed_at: Optional[str] = None
    license: Optional[str] = None  # SPDX name, e.g. "MIT"
    archived: bool = False
    fork: bool = False


class RepositorySearchResult(BaseModel):
    """One page of GitHub repository search results."""

    query: str  # the user's original query, preserved verbatim
    repositories: list[GitHubRepository] = Field(default_factory=list)
    total_count: int = 0
    page: int = 1
    per_page: int = 10
    has_more: bool = False


class TrendingRepository(GitHubRepository):
    """A discovered repository with Northern Star ranking metadata.

    ``rank_change`` is always None until M8.2 introduces stored snapshots —
    M8.1 must not pretend to know history.
    """

    rank: int = 0  # 1-based position in this response
    trend_score: float = 0.0  # deterministic, see discovery service docs
    rank_change: Optional[int] = None


class TrendingResult(BaseModel):
    """Northern Star's API-derived discovery ranking (not an official GitHub ranking)."""

    repositories: list[TrendingRepository] = Field(default_factory=list)
    total: int = 0
    limit: int = 100
    generated_at: Optional[str] = None  # UTC ISO timestamp


# ---------------------------------------------------------------------------
# M8.2 — Historical Trend Intelligence (stored discovery snapshots +
# deterministic growth comparison; nulls wherever history is unavailable)
# ---------------------------------------------------------------------------


class TrendSnapshotResult(BaseModel):
    """Metadata for one explicit snapshot capture (POST /discover/snapshots)."""

    snapshot_at: str  # UTC ISO-8601 of this capture
    limit: int
    repositories_captured: int  # repos returned by discovery for this capture
    new_rows: int  # rows actually inserted
    skipped_duplicates: int  # rows skipped via UNIQUE(full_name, snapshot_at)


class SnapshotPoint(BaseModel):
    """One stored snapshot row for a repository (actual DB content only)."""

    full_name: str
    snapshot_at: str
    rank: Optional[int] = None
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    watchers: int = 0
    pushed_at: Optional[str] = None
    trend_score: Optional[float] = None
    language: Optional[str] = None
    topics: list[str] = Field(default_factory=list)
    html_url: Optional[str] = None


class TrendRepository(BaseModel):
    """Current vs previous comparison for one repository over a window.

    rank_change = previous_rank - current_rank, so positive means the
    repository moved UP, negative means it moved DOWN, zero means unchanged.
    Every historical field is null when history is unavailable — never
    fabricated. ``history_available`` tells the frontend whether to render
    growth badges / rank indicators for this row.
    """

    full_name: str
    html_url: Optional[str] = None
    language: Optional[str] = None
    stars: int = 0
    forks: int = 0
    current_rank: Optional[int] = None
    previous_rank: Optional[int] = None
    rank_change: Optional[int] = None
    previous_stars: Optional[int] = None
    star_delta: Optional[int] = None
    star_growth_percent: Optional[float] = None
    previous_forks: Optional[int] = None
    fork_delta: Optional[int] = None
    fork_growth_percent: Optional[float] = None
    current_trend_score: Optional[float] = None
    previous_trend_score: Optional[float] = None
    trend_score_change: Optional[float] = None
    pushed_at: Optional[str] = None
    previous_pushed_at: Optional[str] = None
    emerging_score: Optional[float] = None
    history_available: bool = False
    comparison_window: str = "7d"


class TrendResult(BaseModel):
    """Windowed trend comparison across the discovery set."""

    window: str  # 24h | 7d | 30d
    generated_at: str
    has_history: bool = False
    history_reason: Optional[str] = None  # why history is unavailable, if so
    current_snapshot_at: Optional[str] = None
    previous_snapshot_at: Optional[str] = None
    repositories: list[TrendRepository] = Field(default_factory=list)
    total: int = 0
    limit: int = 20


class RepositoryHistoryResult(BaseModel):
    """Stored snapshots for one repository plus an optional window comparison."""

    full_name: str
    window: str
    snapshots: list[SnapshotPoint] = Field(default_factory=list)
    total_snapshots: int = 0
    has_history: bool = False
    comparison: Optional[TrendRepository] = None


DirectoryNode.model_rebuild()