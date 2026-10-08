"""Northern Star command-line interface (in-process).

Drives the platform's services directly — same storage, SQLite evidence DB,
and local Ollama as the server — so no HTTP server needs to be running. Zero
extra dependencies (stdlib ``argparse`` only).

Usage (from ``backend/``):

    python -m app.cli ingest https://github.com/pallets/flask
    python -m app.cli manifest pallets/flask
    python -m app.cli index pallets/flask
    python -m app.cli search pallets/flask "routing" --limit 5
    python -m app.cli ask pallets/flask "How does routing work?" --top-k 5
    python -m app.cli version

Exit codes: 0 success, 1 runtime/service error, 2 usage error.
"""

from __future__ import annotations

import json
import sys
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

from .config import get_settings
from .main import APP_VERSION
from .models.schemas import AnswerResponse, ChallengeResult, Claim, IndexSummary, JudgeResult, RepositoryManifest, SearchResponse
from .services import github as github_service
from .services import qa as qa_service
from .services import claims as claims_service
from .services import judge as judge_service
from .services import challenges as challenges_service
from .services.indexing import evidence_db_path, index_repository
from .services.ingestion import ingest_github_repo, load_manifest
from .services.llm import OllamaError
from .services.retrieval import RepoNotIndexedError, search_evidence

# Every expected runtime failure; mapped to a friendly message + exit 1.
_SERVICE_ERRORS = (
    github_service.InvalidGitHubUrlError,
    github_service.RepoNotFoundError,
    github_service.RepoFetchTimeoutError,
    github_service.RepoFetchError,
    RepoNotIndexedError,
    OllamaError,  # base of OllamaUnavailable/Timeout/Response/ModelNotInstalled
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _wrap_errors(fn: Callable) -> Callable:
    """Turn expected service failures into ``error: …`` + exit code 1."""

    @wraps(fn)
    def wrapper(args) -> int:
        try:
            return fn(args)
        except _SERVICE_ERRORS as exc:
            return _fail(str(exc))

    return wrapper


def _parse_owner_repo(value: str) -> tuple[str, str]:
    """Split and validate an ``owner/repo`` positional. Raises ValueError."""
    owner, sep, repo = value.partition("/")
    if not sep or not owner or not repo:
        raise ValueError(f"expected owner/repo, got {value!r}")
    if not github_service.is_valid_slug(owner) or not github_service.is_valid_slug(repo):
        raise ValueError(f"invalid owner/repo slug: {value!r}")
    return owner.lower(), repo.lower()


def _repo_common(value: str) -> tuple[Path, object, str]:
    """Resolve (repo_dir, settings, repo_id) for an owner/repo argument."""
    owner, repo = _parse_owner_repo(value)
    settings = get_settings()
    return settings.storage_root / owner / repo, settings, f"{owner}/{repo}"


def _emit_json(payload: dict) -> int:
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _file_kinds(repo_dir: Path, settings) -> dict[str, str]:
    """Map relative file path → manifest ``FileKind`` for citation labelling.

    Uses the persisted manifest's deterministic file inventory — never a guess.
    Missing/unparseable manifest → empty map (citations render unlabelled).
    """
    raw = load_manifest(repo_dir, settings.manifest_filename)
    if raw is None:
        return {}
    try:
        manifest = RepositoryManifest.model_validate(raw)
    except Exception:
        return {}
    return {f.path: f.category.value for f in manifest.file_inventory}


def _kind_tag(kinds: dict[str, str], file_path: str) -> str:
    """Rendered evidence-type tag for one citation.

    source → [code]; documentation → [documentation]; other FileKinds render
    their literal kind ([config], [data], …). Unknown path → no tag: we never
    label a kind we have no metadata for.
    """
    kind = kinds.get(file_path)
    if kind is None:
        return ""
    if kind == "source":
        return " [code]"
    if kind == "documentation":
        return " [documentation]"
    return f" [{kind}]"


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


@_wrap_errors
def _cmd_ingest(args) -> int:
    manifest = ingest_github_repo(args.url, get_settings(), ref=args.ref)
    if args.json:
        return _emit_json(RepositoryManifest.model_validate(manifest).model_dump(mode="json"))
    langs = ", ".join(f"{s.language}={s.file_count}" for s in manifest.languages)
    fw = ", ".join(f.name for f in manifest.frameworks) or "none"
    print(f"id:          {manifest.id}")
    print(f"url:         {manifest.github_url}")
    print(f"commit:      {manifest.commit_hash or 'n/a'} ({manifest.default_branch or '?'})")
    print(f"files:       {manifest.total_files} total, {manifest.source_files} source")
    if langs:
        print(f"languages:   {langs}")
    print(f"frameworks:  {fw}")
    if manifest.warnings:
        print(f"warnings ({len(manifest.warnings)}):")
        for w in manifest.warnings:
            print(f"  - {w}")
    return 0


@_wrap_errors
def _cmd_manifest(args) -> int:
    repo_dir, settings, repo_id = _repo_common(args.repo)
    raw = load_manifest(repo_dir, settings.manifest_filename)
    if raw is None:
        return _fail("Repository has not been ingested yet.")
    manifest = RepositoryManifest.model_validate(raw)
    if args.json:
        return _emit_json(manifest.model_dump(mode="json"))
    print(f"id:          {manifest.id}")
    print(f"ingested:    {manifest.ingested_at.isoformat()}")
    print(f"commit:      {manifest.commit_hash or 'n/a'} ({manifest.default_branch or '?'})")
    print(f"files:       {manifest.total_files} total, {manifest.source_files} source")
    print(f"size:        {manifest.total_size_bytes} bytes")
    if manifest.languages:
        langs = ", ".join(f"{s.language}={s.file_count}" for s in manifest.languages)
        print(f"languages:   {langs}")
    if manifest.frameworks:
        fw = ", ".join(f"{f.name} ({f.kind})" for f in manifest.frameworks)
        print(f"frameworks:  {fw}")
    if manifest.readme_claims:
        print(f"readme claims ({len(manifest.readme_claims)}):")
        for claim in manifest.readme_claims:
            print(f"  - {claim}")
    if manifest.warnings:
        print(f"warnings ({len(manifest.warnings)}):")
        for w in manifest.warnings:
            print(f"  - {w}")
    return 0


@_wrap_errors
def _cmd_index(args) -> int:
    repo_dir, settings, repo_id = _repo_common(args.repo)
    raw = load_manifest(repo_dir, settings.manifest_filename)
    if raw is None:
        return _fail("Repository has not been ingested yet.")
    manifest = RepositoryManifest.model_validate(raw)
    stats = index_repository(
        manifest,
        repo_dir / "checkout",
        evidence_db_path(settings.storage_root, settings.db_filename),
        chunk_lines=settings.chunk_lines,
    )
    summary = IndexSummary(
        repo_id=stats.repo_id, files_indexed=stats.files_indexed, chunks_created=stats.chunks_created
    )
    if args.json:
        return _emit_json(summary.model_dump(mode="json"))
    print(f"repo:          {summary.repo_id}")
    print(f"files indexed: {summary.files_indexed}")
    print(f"chunks:        {summary.chunks_created}")
    return 0


@_wrap_errors
def _cmd_search(args) -> int:
    repo_dir, settings, repo_id = _repo_common(args.repo)
    if not (repo_dir / settings.manifest_filename).exists():
        return _fail("Repository has not been ingested yet.")
    results = search_evidence(
        evidence_db_path(settings.storage_root, settings.db_filename),
        repo_id,
        args.query,
        limit=args.limit,
        default_limit=settings.search_limit,
    )
    if args.json:
        resp = SearchResponse(
            query=args.query, repo_id=repo_id, total=len(results), results=results
        )
        return _emit_json(resp.model_dump(mode="json"))
    print(f"query: {args.query}")
    print(f"repo:  {repo_id} — {len(results)} result(s)")
    for r in results:
        loc = f"{r.file_path}:{r.start_line}-{r.end_line}"
        lang = f"[{r.language}]" if r.language else "[?]"
        print(f"  {loc}  {lang}  score={r.score}")
        for line in r.content.splitlines()[:6]:
            print(f"      {line[:120]}")
        if len(r.content.splitlines()) > 6:
            print(f"      … ({len(r.content.splitlines()) - 6} more lines)")
    return 0


@_wrap_errors
def _cmd_ask(args) -> int:
    repo_dir, settings, repo_id = _repo_common(args.repo)
    if not (repo_dir / settings.manifest_filename).exists():
        return _fail("Repository has not been ingested yet.")
    if args.model:
        # Settings is a frozen dataclass; replace just the model field.
        from dataclasses import replace

        settings = replace(settings, ollama_model=args.model)
    answer = qa_service.answer_question(
        repo_id, args.question, settings=settings, top_k=args.top_k
    )
    if args.json:
        return _emit_json(AnswerResponse.model_validate(answer).model_dump(mode="json"))
    kinds = _file_kinds(repo_dir, settings) if answer.citations else {}
    # "high" is what the LLM claims, never proof that Northern Star verified it.
    src_label = "model-reported" if answer.confidence_source == "model" else "deterministic"
    print(answer.answer)
    print()
    print(f"confidence: {answer.confidence} ({src_label})")
    print(f"evidence_sufficient: {answer.evidence_sufficient}")
    print(f"evidence_grounding: {answer.evidence_grounding}")
    print(f"evidence_blocks: {len(answer.citations)}")
    if answer.citations:
        print("citations:")
        for c in answer.citations:
            print(f"  {c.id}  {c.file_path}:{c.start_line}-{c.end_line}{_kind_tag(kinds, c.file_path)}")
    else:
        print("citations: (none)")
    return 0


@_wrap_errors
def _cmd_claims(args) -> int:
    repo_dir, settings, repo_id = _repo_common(args.repo)
    raw = load_manifest(repo_dir, settings.manifest_filename)
    if raw is None:
        return _fail("Repository has not been ingested yet.")
    manifest = RepositoryManifest.model_validate(raw)
    if not manifest.readme_present:
        if args.json:
            return _emit_json([])
        print("No README present.")
        return 0
    # Read README content and extract structured claims using the new extraction
    readme_path = repo_dir / "checkout" / "README.md"
    if not readme_path.exists():
        # Try case-insensitive
        for f in (repo_dir / "checkout").iterdir():
            if f.name.lower() == "readme.md":
                readme_path = f
                break
    if not readme_path.exists():
        if args.json:
            return _emit_json([])
        print("README.md not found in checkout.")
        return 0
    readme_text = readme_path.read_text(encoding="utf-8", errors="replace")
    from app.services.detection import extract_structured_claims
    claims_list = extract_structured_claims(readme_text, source="README.md")
    # Set repo_id on each claim
    for c in claims_list:
        c.repo_id = repo_id
    if args.json:
        return _emit_json([c.model_dump(mode="json") for c in claims_list])
    print(f"Repository: {repo_id}")
    print(f"README claims found: {len(claims_list)}")
    for c in claims_list:
        print(f"  {c.id}: {c.text} [category: {c.category}]")
    return 0


@_wrap_errors
def _cmd_verify(args) -> int:
    repo_dir, settings, repo_id = _repo_common(args.repo)
    if not (repo_dir / settings.manifest_filename).exists():
        return _fail("Repository has not been ingested yet.")
    if args.model:
        from dataclasses import replace

        settings = replace(settings, ollama_model=args.model)

    claim = Claim(
        id="claim_0",
        text=args.claim,
        source=args.source,
        kind=args.kind,
        category=args.category,
        verdict="unclear",
        verdict_explanation="",
        evidence_ids=[],
        repo_id=repo_id,
    )
    verified = claims_service.verify_claim(
        repo_id=repo_id,
        claim=claim,
        settings=settings,
        top_k=args.top_k,
    )
    if args.json:
        return _emit_json(Claim.model_validate(verified).model_dump(mode="json"))
    print(f"repo:    {repo_id}")
    print(f"claim:   {verified.text}")
    print(f"source:  {verified.source} ({verified.kind})")
    print(f"verdict: {verified.verdict}")
    print()
    print("explanation:")
    print(verified.verdict_explanation)
    print()
    if verified.evidence_ids:
        kinds = _file_kinds(repo_dir, settings)
        print("evidence:")
        for eid in verified.evidence_ids:
            # We don't have the full citation objects here, but we can show the IDs
            print(f"  {eid}")
    else:
        print("evidence: (none)")
    return 0


def _cmd_version(args) -> int:
    print(APP_VERSION)
    return 0


@_wrap_errors
def _cmd_judge(args) -> int:
    repo_dir, settings, repo_id = _repo_common(args.repo)
    if not (repo_dir / settings.manifest_filename).exists():
        return _fail("Repository has not been ingested yet.")
    if args.model:
        from dataclasses import replace

        settings = replace(settings, ollama_model=args.model)

    judged = judge_service.judge_repository(
        repo_id=repo_id,
        settings=settings,
        top_k=args.top_k,
    )
    if args.json:
        return _emit_json(JudgeResult.model_validate(judged).model_dump(mode="json"))

    print(f"repo:    {repo_id}")
    print(f"overall score: {judged.overall_score}/100")
    print()
    print("dimensions:")
    for dim in judged.dimensions:
        print(f"  {dim.name}: {dim.score}/10")
        if dim.explanation:
            print(f"    {dim.explanation}")
    print()
    print("strengths:")
    for s in judged.strengths:
        print(f"  + {s}")
    print()
    print("weaknesses:")
    for w in judged.weaknesses:
        print(f"  - {w}")
    print()
    print("recommendations:")
    for r in judged.recommendations:
        print(f"  > {r}")
    print()
    print("claim integrity:")
    for verdict, count in judged.claim_integrity_summary.items():
        print(f"  {verdict}: {count}")
    print(f"  total claims: {judged.total_claims}")
    print()
    if judged.evidence_citations:
        print("evidence:")
        for c in judged.evidence_citations:
            print(f"  {c.id}  {c.file_path}:{c.start_line}-{c.end_line}")
    else:
        print("evidence: (none)")
    return 0


@_wrap_errors
def _cmd_challenges(args) -> int:
    repo_dir, settings, repo_id = _repo_common(args.repo)
    if not (repo_dir / settings.manifest_filename).exists():
        return _fail("Repository has not been ingested yet.")
    if args.model:
        from dataclasses import replace

        settings = replace(settings, ollama_model=args.model)

    result = challenges_service.generate_challenges(
        repo_id=repo_id,
        settings=settings,
        top_k=args.top_k,
    )
    if args.json:
        return _emit_json(ChallengeResult.model_validate(result).model_dump(mode="json"))

    print(f"repo:    {repo_id}")
    print(f"total challenges: {result.total_challenges}")
    print(f"  high: {result.high_severity}, medium: {result.medium_severity}, low: {result.low_severity}")
    print()
    for c in result.challenges:
        print(f"challenge {c.id}:")
        print(f"  claim:     {c.claim}")
        print(f"  challenge: {c.challenge}")
        print(f"  severity:  {c.severity}")
        print(f"  category:  {c.category}")
        print(f"  confidence: {c.confidence}")
        print(f"  explanation: {c.explanation}")
        if c.evidence_ids:
            print(f"  evidence:  {', '.join(c.evidence_ids)}")
        print()
    print("claim integrity summary:")
    print()
    if result.evidence_citations:
        print("evidence:")
        for c in result.evidence_citations:
            print(f"  {c.id}  {c.file_path}:{c.start_line}-{c.end_line}")
    else:
        print("evidence: (none)")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="northernstar",
        description="Northern Star — software intelligence, from the terminal (in-process).",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def common(p):
        p.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    p = sub.add_parser("ingest", help="clone + analyse + index a GitHub repository")
    p.add_argument("url", help="https://github.com/owner/repo")
    p.add_argument("--ref", default=None, help="reserved; default branch is cloned")
    common(p)
    p.set_defaults(handler=_cmd_ingest)

    p = sub.add_parser("manifest", help="show the ingested metadata report")
    p.add_argument("repo", metavar="owner/repo")
    common(p)
    p.set_defaults(handler=_cmd_manifest)

    p = sub.add_parser("index", help="build (or rebuild) the evidence index")
    p.add_argument("repo", metavar="owner/repo")
    common(p)
    p.set_defaults(handler=_cmd_index)

    p = sub.add_parser("search", help="search the repo's evidence index (lexical, FTS5)")
    p.add_argument("repo", metavar="owner/repo")
    p.add_argument("query", help="free-text search query")
    p.add_argument("--limit", type=int, default=None, help="max results (default: NORTHERN_STAR_SEARCH_LIMIT)")
    common(p)
    p.set_defaults(handler=_cmd_search)

    p = sub.add_parser("ask", help="ask a question grounded in the repo's evidence")
    p.add_argument("repo", metavar="owner/repo")
    p.add_argument("question", help='question (quote it), e.g. "How does routing work?"')
    p.add_argument("--top-k", type=int, default=None, help="evidence chunks used (default: QA_TOP_K)")
    p.add_argument("--model", default=None, help="Ollama model for this ask (default: OLLAMA_MODEL)")
    common(p)
    p.set_defaults(handler=_cmd_ask)

    p = sub.add_parser("claims", help="extract structured claims from README")
    p.add_argument("repo", metavar="owner/repo")
    common(p)
    p.set_defaults(handler=_cmd_claims)

    p = sub.add_parser("verify", help="verify a claim against the repo's evidence")
    p.add_argument("repo", metavar="owner/repo")
    p.add_argument("claim", help="claim text to verify (quote it)")
    p.add_argument("--source", default="README.md", help="source file path for provenance")
    p.add_argument("--kind", default="documentation", help="file kind: source, documentation, config, etc.")
    p.add_argument("--category", default="general", help="claim category")
    p.add_argument("--top-k", type=int, default=None, help="evidence chunks used (default: QA_TOP_K)")
    p.add_argument("--model", default=None, help="Ollama model for this verify (default: OLLAMA_MODEL)")
    common(p)
    p.set_defaults(handler=_cmd_verify)

    p = sub.add_parser("judge", help="judge a repository across five dimensions using evidence")
    p.add_argument("repo", metavar="owner/repo")
    p.add_argument("--top-k", type=int, default=None, help="evidence chunks used per dimension (default: QA_TOP_K)")
    p.add_argument("--model", default=None, help="Ollama model for this judge (default: OLLAMA_MODEL)")
    common(p)
    p.set_defaults(handler=_cmd_judge)

    p = sub.add_parser("challenges", help="generate red-team challenges for a repository")
    p.add_argument("repo", metavar="owner/repo")
    p.add_argument("--top-k", type=int, default=None, help="evidence chunks used per dimension (default: QA_TOP_K)")
    p.add_argument("--model", default=None, help="Ollama model for this challenges (default: OLLAMA_MODEL)")
    common(p)
    p.set_defaults(handler=_cmd_challenges)

    p = sub.add_parser("version", help="print the version")
    p.set_defaults(handler=_cmd_version)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse already printed the usage/error; translate to our exit code.
        return int(exc.code or 0)
    if not hasattr(args, "handler"):
        parser.print_help(sys.stderr)
        return 2
    try:
        return args.handler(args)
    except ValueError as exc:  # bad owner/repo argument
        return _fail(str(exc))


if __name__ == "__main__":
    sys.exit(main())