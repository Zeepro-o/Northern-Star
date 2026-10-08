"""GitHub URL validation and repository fetching via the git CLI.

Kept as a service separate from the route handlers so later milestones
(indexing, Q&A, judging) can reuse the same clone/storage layout.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

GITHUB_HTTP_RE = re.compile(
    r"^https://(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
SHORTHAND_RE = re.compile(r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)$")
SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

_DISALLOWED_SLUGS = {".", ".."}


def is_valid_slug(value: str) -> bool:
    return bool(SLUG_RE.fullmatch(value)) and value not in _DISALLOWED_SLUGS


class InvalidGitHubUrlError(ValueError):
    """The URL is not a shape we are willing to fetch."""


class RepoNotFoundError(Exception):
    """The repository does not exist or is not accessible (private)."""


class RepoFetchTimeoutError(Exception):
    """A git operation exceeded the configured timeout."""


class RepoFetchError(Exception):
    """Cloning or talking to the remote failed for another reason."""


@dataclass(frozen=True)
class GitHubRepo:
    owner: str
    repo: str
    canonical_url: str


@dataclass(frozen=True)
class FetchedRepository:
    owner: str
    repo: str
    github_url: str
    checkout_root: Path
    commit_hash: Optional[str] = None
    default_branch: Optional[str] = None


def parse_github_url(url: str) -> GitHubRepo:
    """Parse a user-supplied GitHub URL.

    Accepted:
      * https://github.com/owner/repo
      * https://github.com/owner/repo.git
      * owner/repo (shorthand)

    Rejected:
      * http:// (insecure)
      * ssh:// or git@... (credentials are not supported in v0.1)
      * any non-github.com host
    """
    if not url or not url.strip():
        raise InvalidGitHubUrlError("URL must not be empty.")

    url = url.strip()

    if url.startswith("git@") or url.startswith("ssh://"):
        raise InvalidGitHubUrlError(
            "SSH URLs are not supported in v0.1; use https://github.com/owner/repo."
        )
    if "://" in url:
        scheme = url.split("://", 1)[0]
        if scheme != "https":
            raise InvalidGitHubUrlError("Only https:// URLs are accepted.")

    m = GITHUB_HTTP_RE.match(url)
    if m:
        owner, repo = m.group("owner"), m.group("repo")
        if not (is_valid_slug(owner) and is_valid_slug(repo)):
            raise InvalidGitHubUrlError(f"URL contains an invalid owner/repo: {owner}/{repo}")
        return GitHubRepo(owner=owner, repo=repo, canonical_url=f"https://github.com/{owner}/{repo}")

    m = SHORTHAND_RE.match(url)
    if m and "://" not in url:
        owner, repo = m.group("owner"), m.group("repo")
        if is_valid_slug(owner) and is_valid_slug(repo):
            return GitHubRepo(owner=owner, repo=repo, canonical_url=f"https://github.com/{owner}/{repo}")

    raise InvalidGitHubUrlError(
        "Expected https://github.com/owner/repo (or the 'owner/repo' shorthand)."
    )


def _run_git(args: list[str], *, cwd: Optional[Path] = None, timeout: int) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "echo"}
    try:
        proc = subprocess.run(
            args, cwd=str(cwd) if cwd else None, env=env,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RepoFetchTimeoutError(
            f"git {args[0]} timed out after {timeout}s"
        ) from exc
    if proc.returncode != 0:
        raise RepoFetchError(f"git {args[0]} failed: {proc.stderr.strip()[:400]}")
    return proc


def fetch_repository(url: str, base_dir: Path, timeout: int) -> FetchedRepository:
    """Validate the URL, shallow-clone the repository, and return clone metadata.

    Storage layout on disk: ``<base_dir>/owner/repo/checkout``.
    """
    repo = parse_github_url(url)
    repo_dir = base_dir / repo.owner.lower() / repo.repo.lower()
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)
    checkout = repo_dir / "checkout"

    # Preflight: ls-remote fails quickly (404) for private/missing repos and
    # costs far less than a full clone. GIT_TERMINAL_PROMPT=0 prevents hangs
    # on credential prompts for private repos.
    try:
        _run_git(["git", "ls-remote", "--heads", repo.canonical_url], timeout=timeout)
    except RepoFetchTimeoutError:
        raise
    except RepoFetchError as exc:
        shutil.rmtree(repo_dir, ignore_errors=True)
        raise RepoNotFoundError(
            f"Repository not found or not accessible (private repos are not supported in v0.1): "
            f"{repo.canonical_url}. {exc}"
        ) from exc

    try:
        _run_git(
            ["git", "clone", "--depth", "1", "--quiet", "--", repo.canonical_url, str(checkout)],
            timeout=timeout,
        )
    except RepoFetchTimeoutError:
        shutil.rmtree(repo_dir, ignore_errors=True)
        raise
    except RepoFetchError as exc:
        shutil.rmtree(repo_dir, ignore_errors=True)
        raise RepoFetchError(
            f"Unable to clone repository '{repo.canonical_url}' — it may be empty or in an "
            f"unsupported state. ({exc})"
        ) from exc

    branch: Optional[str] = None
    commit: Optional[str] = None
    try:
        proc = _run_git(["git", "-C", str(checkout), "rev-parse", "--abbrev-ref", "HEAD"], timeout=timeout)
        found = proc.stdout.strip()
        branch = found if found and found != "HEAD" else None
        proc = _run_git(["git", "-C", str(checkout), "rev-parse", "HEAD"], timeout=timeout)
        commit = proc.stdout.strip() or None
    except RepoFetchError:
        pass  # Degenerate checkout (used in tests); provenance fields simply stay empty.

    return FetchedRepository(
        owner=repo.owner,
        repo=repo.repo,
        github_url=repo.canonical_url,
        checkout_root=checkout,
        commit_hash=commit,
        default_branch=branch,
    )