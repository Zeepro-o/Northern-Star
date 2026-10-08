"""HTTP API routes for Northern Star.

v0.1: repository ingestion endpoint.
v0.2: evidence-index (re)build + search endpoints.
v0.3: evidence-grounded Q&A endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..config import get_settings
from ..models.schemas import (
    AnswerResponse,
    IndexSummary,
    QuestionRequest,
    RepositoryManifest,
    RepositoryRequest,
    SearchResponse,
)
from ..services import github as github_service
from ..services import qa as qa_service
from ..services.indexing import evidence_db_path, index_repository
from ..services.ingestion import ingest_github_repo, load_manifest
from ..services.llm import (
    OllamaModelNotInstalledError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from ..services.retrieval import RepoNotIndexedError, search_evidence

router = APIRouter(tags=["repositories"])


def _repo_dir(owner: str, repo: str):
    """Validate slugs and return their storage directory (or raise 400)."""
    if not github_service.is_valid_slug(owner) or not github_service.is_valid_slug(repo):
        raise HTTPException(status_code=400, detail="Invalid owner/repo slug.")
    settings = get_settings()
    return settings.storage_root / owner.lower() / repo.lower(), settings


@router.post(
    "/repos",
    response_model=RepositoryManifest,
    status_code=201,
    summary="Ingest a GitHub repository and return its structured metadata report",
)
def ingest_repository(payload: RepositoryRequest) -> RepositoryManifest:
    settings = get_settings()
    try:
        return ingest_github_repo(payload.url, settings, ref=payload.ref)
    except github_service.InvalidGitHubUrlError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid repository URL: {exc}")
    except github_service.RepoNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except github_service.RepoFetchTimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Timed out while fetching the repository from the remote host.",
        )
    except github_service.RepoFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get(
    "/repos/{owner}/{repo}",
    response_model=RepositoryManifest,
    summary="Retrieve a previously ingested repository manifest",
)
def get_repository_manifest(owner: str, repo: str) -> RepositoryManifest:
    repo_dir, settings = _repo_dir(owner, repo)
    manifest = load_manifest(repo_dir, settings.manifest_filename)
    if manifest is None:
        raise HTTPException(
            status_code=404, detail="Repository has not been ingested yet."
        )
    return RepositoryManifest.model_validate(manifest)


@router.post(
    "/repos/{owner}/{repo}/index",
    response_model=IndexSummary,
    summary="Build (or rebuild) the SQLite evidence index for a repository",
)
def index_repository_endpoint(owner: str, repo: str) -> IndexSummary:
    repo_dir, settings = _repo_dir(owner, repo)
    raw = load_manifest(repo_dir, settings.manifest_filename)
    if raw is None:
        raise HTTPException(
            status_code=404, detail="Repository has not been ingested yet."
        )
    manifest = RepositoryManifest.model_validate(raw)
    stats = index_repository(
        manifest,
        repo_dir / "checkout",
        evidence_db_path(settings.storage_root, settings.db_filename),
        chunk_lines=settings.chunk_lines,
    )
    return IndexSummary(
        repo_id=stats.repo_id,
        files_indexed=stats.files_indexed,
        chunks_created=stats.chunks_created,
    )


@router.get(
    "/repos/{owner}/{repo}/search",
    response_model=SearchResponse,
    summary="Search a repository's evidence index (lexical, FTS5)",
)
def search_repository(
    owner: str,
    repo: str,
    q: str = Query(..., min_length=1, description="Search query (free text)"),
    limit: int = Query(None, ge=1, le=100, description="Max results (default 20)"),
) -> SearchResponse:
    repo_dir, settings = _repo_dir(owner, repo)
    if not (repo_dir / settings.manifest_filename).exists():
        raise HTTPException(
            status_code=404, detail="Repository has not been ingested yet."
        )
    repo_id = f"{owner.lower()}/{repo.lower()}"
    db_path = evidence_db_path(settings.storage_root, settings.db_filename)
    try:
        results = search_evidence(
            db_path,
            repo_id,
            q,
            limit=limit,
            default_limit=settings.search_limit,
        )
    except RepoNotIndexedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return SearchResponse(
        query=q,
        repo_id=repo_id,
        total=len(results),
        results=results,
    )


@router.post(
    "/repos/{owner}/{repo}/ask",
    response_model=AnswerResponse,
    summary=(
        "Ask a question about a repository, answered from its evidence index "
        "with validated file:line citations"
    ),
)
def ask_repository_query(
    owner: str, repo: str, payload: QuestionRequest
) -> AnswerResponse:
    repo_dir, settings = _repo_dir(owner, repo)
    if not (repo_dir / settings.manifest_filename).exists():
        raise HTTPException(
            status_code=404, detail="Repository has not been ingested yet."
        )
    repo_id = f"{owner.lower()}/{repo.lower()}"
    try:
        return qa_service.answer_question(
            repo_id,
            payload.question,
            settings=settings,
            top_k=payload.top_k,
        )
    except RepoNotIndexedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (OllamaUnavailableError, OllamaModelNotInstalledError) as exc:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {exc}")
    except OllamaTimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Ollama timed out while generating the answer.",
        )
    except OllamaResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc))