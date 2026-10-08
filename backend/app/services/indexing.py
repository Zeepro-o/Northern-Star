"""SQLite evidence index for Northern Star M2.

Responsibilities (kept layered, per milestone plan):

    ingestion → indexing service → SQLite evidence database

The index is a single SQLite database (``<storage_root>/northern_star.db``)
holding repositories → files → chunks, with an FTS5 virtual table over chunk
content for lexical retrieval. Re-indexing the same repository is idempotent:
prior rows for that repo are deleted inside one transaction before inserts.

Deliberately built only on Python's built-in ``sqlite3`` module — no native
dependencies beyond what ships with Python 3.14.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..models.schemas import FileKind, RepositoryManifest
from .chunking import chunk_file

# Files classified into these categories become searchable evidence.
INDEXABLE_CATEGORIES: set[FileKind] = {
    FileKind.SOURCE,
    FileKind.CONFIG,
    FileKind.DOCUMENTATION,
    FileKind.DATA,
}

# Files larger than this are skipped by the indexer (large binaries or
# generated blobs that would dominate chunk counts without helping retrieval).
MAX_INDEX_FILE_BYTES: int = 1024 * 1024

# Default small-file threshold (a file this many lines or fewer stays one chunk).
MAX_LINES_SINGLE_CHUNK: int = 150

SCHEMA = """
CREATE TABLE IF NOT EXISTS repositories (
    id TEXT PRIMARY KEY,               -- "owner/repo"
    commit_hash TEXT,
    default_branch TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id TEXT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    path TEXT NOT NULL,                -- relative to repo root (citation anchor)
    language TEXT,
    size_bytes INTEGER,
    is_source INTEGER NOT NULL DEFAULT 0,
    UNIQUE(repo_id, path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id TEXT NOT NULL,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    language TEXT,
    start_line INTEGER NOT NULL,       -- 1-indexed, inclusive
    end_line INTEGER NOT NULL,         -- 1-indexed, inclusive
    content TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_repo ON files(repo_id);
CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id);

-- External-content FTS5 index over chunk text. Triggers keep it in sync with
-- the chunks table so we never need a manual "rebuild" on normal writes.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


@dataclass(frozen=True)
class IndexStats:
    """Result summary of one (re-)indexing operation."""

    repo_id: str
    files_indexed: int
    chunks_created: int


def evidence_db_path(storage_root: Path, db_filename: str) -> Path:
    """Where the evidence SQLite database lives."""
    return storage_root / db_filename


def connect_evidence_db(db_path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the evidence database with FKs enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def repo_is_indexed(db_path: Path, repo_id: str) -> bool:
    """True if the repository is present in the evidence index."""
    with connect_evidence_db(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM repositories WHERE id = ?", (repo_id,)
        ).fetchone()
        return row is not None


def _read_text_safe(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def index_repository(
    manifest: RepositoryManifest,
    checkout_root: Path,
    db_path: Path,
    *,
    chunk_lines: int = 100,
    max_lines: int = MAX_LINES_SINGLE_CHUNK,
    max_file_bytes: int = MAX_INDEX_FILE_BYTES,
) -> IndexStats:
    """Build (or rebuild) the evidence index for one repository.

    Idempotent: any previously indexed rows for ``manifest.id`` are removed in
    the same transaction as the new inserts, so re-indexing never duplicates.
    """
    repo_id = manifest.id
    conn = connect_evidence_db(db_path)
    try:
        with conn:
            # Clear stale rows first (chunks → files → repository). The chunk
            # DELETE fires chunks_ad, which removes the matching FTS rows.
            conn.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM files WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM repositories WHERE id = ?", (repo_id,))

            conn.execute(
                "INSERT INTO repositories (id, commit_hash, default_branch, created_at)"
                " VALUES (?, ?, ?, ?)",
                (
                    repo_id,
                    manifest.commit_hash,
                    manifest.default_branch,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            files_indexed = 0
            chunks_created = 0
            for entry in sorted(manifest.file_inventory, key=lambda e: e.path):
                if entry.category not in INDEXABLE_CATEGORIES:
                    continue
                source = checkout_root / entry.path
                if not source.is_file() or entry.size_bytes > max_file_bytes:
                    continue
                content = _read_text_safe(source)
                if content is None:
                    continue

                cur = conn.execute(
                    "INSERT INTO files (repo_id, path, language, size_bytes, is_source)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        repo_id,
                        entry.path,
                        entry.language,
                        entry.size_bytes,
                        1 if entry.category == FileKind.SOURCE else 0,
                    ),
                )
                file_id = cur.lastrowid
                files_indexed += 1

                for chunk in chunk_file(
                    repo_id,
                    entry.path,
                    content,
                    entry.language,
                    chunk_lines=chunk_lines,
                    max_lines=max_lines,
                ):
                    conn.execute(
                        "INSERT INTO chunks"
                        " (repo_id, file_id, file_path, language, start_line, end_line, content)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            chunk.repo_id,
                            file_id,
                            chunk.file_path,
                            chunk.language,
                            chunk.start_line,
                            chunk.end_line,
                            chunk.content,
                        ),
                    )
                    chunks_created += 1
    finally:
        conn.close()
    return IndexStats(repo_id=repo_id, files_indexed=files_indexed, chunks_created=chunks_created)