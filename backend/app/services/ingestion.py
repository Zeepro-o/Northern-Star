"""Repository ingestion: clone → walk → detect → assemble → persist manifest.

Split into a service (not route logic) so future milestones can reuse the same
repository representation for indexing, Q&A, judging and red-teaming.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import Settings
from ..models.schemas import (
    CategoryStats,
    DirectoryNode,
    FileEntry,
    FileKind,
    FrameworkInfo,
    IgnoredDirectory,
    LanguageStats,
    RepositoryManifest,
)
from .detection import (
    _dir_skip_reason,
    _file_skip_reason,
    classify_file,
    detect_frameworks,
    extract_readme_claims,
    is_readme_filename,
)
from .github import FetchedRepository, fetch_repository
from .indexing import evidence_db_path, index_repository


class _LimitReached(Exception):
    """Internal signal: safe-analysis limits were hit; stop walking."""


@dataclass
class Analysis:
    """The deterministic result of analysing one checkout (no git context)."""

    entries: list[FileEntry] = field(default_factory=list)
    languages: list[LanguageStats] = field(default_factory=list)
    categories: list[CategoryStats] = field(default_factory=list)
    frameworks: list[FrameworkInfo] = field(default_factory=list)
    package_manager: Optional[str] = None
    config_files: list[str] = field(default_factory=list)
    readme_present: bool = False
    readme_claims: list[str] = field(default_factory=list)
    ignored_directories: list[IgnoredDirectory] = field(default_factory=list)
    directory_structure: Optional[DirectoryNode] = None
    total_files: int = 0
    source_files: int = 0
    total_size_bytes: int = 0
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)


def _read_text_safe(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _insert_tree(node: DirectoryNode, parts: tuple[str, ...], size: int) -> None:
    node.file_count += 1
    node.size_bytes += size
    current_parts = list(Path(node.path).parts) if node.path else []
    for part in parts:
        current_parts.append(part)
        child = next((c for c in node.children if c.name == part), None)
        if child is None:
            child = DirectoryNode(
                name=part, path="/".join(current_parts), file_count=0, size_bytes=0
            )
            node.children.append(child)
        node = child
        node.file_count += 1
        node.size_bytes += size


def _prune_tree(node: DirectoryNode, depth: int, settings: Settings) -> None:
    if depth >= settings.dir_tree_max_depth and node.children:
        node.children = []
        node.truncated = True
        return
    if len(node.children) > settings.dir_tree_max_children:
        node.children.sort(key=lambda c: -c.size_bytes)
        node.children = node.children[: settings.dir_tree_max_children]
        node.truncated = True
    for child in node.children:
        _prune_tree(child, depth + 1, settings)


def build_directory_tree(entries: list[FileEntry], root_name: str, settings: Settings) -> DirectoryNode:
    root = DirectoryNode(name=root_name, path="", file_count=0, size_bytes=0)
    for entry in entries:
        if not entry.path:
            continue
        parts = Path(entry.path).parent.parts
        _insert_tree(root, parts, entry.size_bytes)
    _prune_tree(root, 0, settings)
    return root


def analyze_repository(checkout: Path, settings: Settings, tree_root_name: str = ".") -> Analysis:
    """Walk a checkout and build deterministic language/framework metadata."""
    analysis = Analysis()
    config_texts: dict[str, str] = {}
    lang_counts: dict[str, int] = {}
    lang_sizes: dict[str, int] = {}
    cat_counts: dict[str, int] = {}
    cat_sizes: dict[str, int] = {}
    readme_candidate: Optional[Path] = None

    try:
        for root, dirs, filenames in os.walk(checkout):
            root_path = Path(root)
            rel_root = root_path.relative_to(checkout)

            kept: list[str] = []
            for d in sorted(dirs):
                reason = _dir_skip_reason(d)
                if reason:
                    analysis.ignored_directories.append(
                        IgnoredDirectory(path=(rel_root / d).as_posix(), reason=reason)
                    )
                else:
                    kept.append(d)
            dirs[:] = kept

            for filename in sorted(filenames):
                if analysis.total_files >= settings.analyze_max_files:
                    analysis.truncated = True
                    analysis.warnings.append(
                        f"Ingestion stopped at MAX_FILES={settings.analyze_max_files}; "
                        "the file inventory is incomplete."
                    )
                    raise _LimitReached

                skip_reason = _file_skip_reason(filename)
                if skip_reason:
                    analysis.warnings.append(f"Skipped '{filename}' ({skip_reason}).")
                    continue

                full = root_path / filename
                if full.is_symlink() or not full.is_file():
                    continue

                rel = (rel_root / filename).as_posix()
                try:
                    size = full.stat().st_size
                except OSError:
                    continue

                language, category = classify_file(full, settings.binary_sniff_bytes)
                entry = FileEntry(
                    path=rel, language=language, category=category, size_bytes=size
                )
                analysis.entries.append(entry)
                analysis.total_files += 1
                analysis.total_size_bytes += size

                if category == FileKind.SOURCE:
                    analysis.source_files += 1

                lang_key = language or "unknown"
                lang_counts[lang_key] = lang_counts.get(lang_key, 0) + 1
                lang_sizes[lang_key] = lang_sizes.get(lang_key, 0) + size

                cat_key = category.value
                cat_counts[cat_key] = cat_counts.get(cat_key, 0) + 1
                cat_sizes[cat_key] = cat_sizes.get(cat_key, 0) + size

                if category == FileKind.CONFIG:
                    analysis.config_files.append(rel)
                    if size <= settings.max_config_file_size_bytes:
                        text = _read_text_safe(full)
                        if text is not None:
                            config_texts[rel] = text

                if rel_root == Path(".") and is_readme_filename(filename):
                    if readme_candidate is None:
                        readme_candidate = full

                if analysis.total_size_bytes > settings.analyze_max_repo_size_bytes:
                    analysis.truncated = True
                    analysis.warnings.append(
                        "Repository total size exceeded MAX_REPO_SIZE_BYTES; analysis may be incomplete."
                    )
                    raise _LimitReached
    except _LimitReached:
        pass  # walk deliberately stopped; results so far are still valid

    if readme_candidate is not None:
        analysis.readme_present = True
        analysis.readme_claims = extract_readme_claims(_read_text_safe(readme_candidate) or "")

    entry_paths = {e.path for e in analysis.entries}
    analysis.frameworks, analysis.package_manager = detect_frameworks(config_texts, entry_paths)

    analysis.languages = [
        LanguageStats(language=lang, file_count=lang_counts[lang], size_bytes=lang_sizes[lang])
        for lang in sorted(lang_counts, key=lambda k: (-lang_counts[k], k))
    ]
    analysis.categories = [
        CategoryStats(category=FileKind(kind), file_count=cat_counts[kind], size_bytes=cat_sizes[kind])
        for kind in sorted(cat_counts, key=lambda k: (-cat_counts[k], k))
    ]
    analysis.config_files = sorted(analysis.config_files)
    analysis.directory_structure = build_directory_tree(analysis.entries, tree_root_name, settings)
    return analysis


def _assemble_manifest(
    fetched: FetchedRepository, analysis: Analysis, settings: Settings, ref: Optional[str]
) -> RepositoryManifest:
    return RepositoryManifest(
        schema_version=settings.schema_version,
        ingested_at=datetime.now(timezone.utc),
        id=f"{fetched.owner.lower()}/{fetched.repo.lower()}",
        owner=fetched.owner,
        repo=fetched.repo,
        github_url=fetched.github_url,
        clone_ref=ref,
        commit_hash=fetched.commit_hash,
        default_branch=fetched.default_branch,
        readme_present=analysis.readme_present,
        readme_claims=analysis.readme_claims,
        total_files=analysis.total_files,
        source_files=analysis.source_files,
        total_size_bytes=analysis.total_size_bytes,
        languages=analysis.languages,
        categories=analysis.categories,
        frameworks=analysis.frameworks,
        package_manager=analysis.package_manager,
        config_files=analysis.config_files,
        ignored_directories=analysis.ignored_directories,
        directory_structure=analysis.directory_structure,
        file_inventory=analysis.entries,
        truncated=analysis.truncated,
        warnings=analysis.warnings
        + [
            "Shallow clone (depth=1); full commit history is not available."
            if fetched.commit_hash is not None
            else "No git metadata found; repository may be temporary or the clone is incomplete."
        ],
    )


def _persist_manifest(fetched: FetchedRepository, manifest: RepositoryManifest, settings: Settings) -> None:
    target = fetched.checkout_root.parent / settings.manifest_filename
    target.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def load_manifest(repo_dir: Path, manifest_filename: str) -> Optional[dict]:
    path = repo_dir / manifest_filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ingest_github_repo(url: str, settings: Settings, ref: Optional[str] = None) -> RepositoryManifest:
    """Full pipeline: validate + clone a GitHub repo, analyse it, persist the
    manifest, and build the SQLite evidence index (M2).

    Indexing failures are recorded as manifest warnings rather than failing the
    ingest: the metadata report is still valid, and the index can be rebuilt
    later via ``POST /repos/{owner}/{repo}/index``.
    """
    fetched = fetch_repository(url, settings.storage_root, settings.clone_timeout_seconds)
    analysis = analyze_repository(fetched.checkout_root, settings, tree_root_name=fetched.repo)
    manifest = _assemble_manifest(fetched, analysis, settings, ref)
    _persist_manifest(fetched, manifest, settings)
    try:
        index_repository(
            manifest,
            fetched.checkout_root,
            evidence_db_path(settings.storage_root, settings.db_filename),
            chunk_lines=settings.chunk_lines,
        )
    except Exception as exc:  # noqa: BLE001 — indexing must not break ingestion
        manifest.warnings.append(f"Evidence index build failed: {exc}")
    return manifest