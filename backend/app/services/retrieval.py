"""Lexical evidence retrieval over the SQLite FTS5 index.

Given a repository and a query, return the most relevant evidence chunks
with exact file + line provenance. Pure SQLite FTS5 + BM25 ranking — no LLM,
no embeddings. This layer is what future milestones (M3+) will cite from.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..models.schemas import SearchResult
from .indexing import connect_evidence_db, repo_is_indexed

# FTS5 tokens are [A-Za-z0-9_]: everything else is a separator for our
# purposes. We re-build a quoted, OR-joined MATCH expression from this so we
# never hand user input to the FTS parser verbatim (no bare "OR"/"NEAR" or
# stray quotes to break the query, and no injection into the MATCH grammar).
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# Returns no rows for any sensible query; used when the user query yields zero
# tokens and we must still produce a valid FTS statement.
_NO_MATCH = '"__northern_star_no_match__"'


class RepoNotIndexedError(LookupError):
    """The repository manifest exists but has no evidence index yet."""


def _fts_match_expression(raw_query: str) -> str:
    tokens = _TOKEN_RE.findall(raw_query or "")
    if not tokens:
        return _NO_MATCH
    return " OR ".join(f'"{tok}"' for tok in tokens)


def _fts_safe_limit(limit: int, default: int = 20, maximum: int = 100) -> int:
    if limit is None:
        return default
    return max(1, min(int(limit), maximum))


def search_evidence(
    db_path: Path,
    repo_id: str,
    query: str,
    *,
    limit: Optional[int] = None,
    default_limit: int = 20,
) -> list[SearchResult]:
    """Search one repository's evidence index.

    Raises ``RepoNotIndexedError`` when the repository has no index rows.
    Empty queries match nothing and return []. Results are ordered by FTS5
    BM25 ranking (best first) and carry full provenance.
    """
    if not repo_is_indexed(db_path, repo_id):
        raise RepoNotIndexedError(f"Repository '{repo_id}' is not indexed yet.")

    match_expr = _fts_match_expression(query)
    limit = _fts_safe_limit(limit, default=default_limit)

    results: list[SearchResult] = []
    with connect_evidence_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.file_path AS file_path,
                   c.start_line  AS start_line,
                   c.end_line    AS end_line,
                   c.language    AS language,
                   c.content     AS content,
                   bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks AS c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ? AND c.repo_id = ?
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            (match_expr, repo_id, limit),
        )
        for row in rows:
            results.append(
                SearchResult(
                    file_path=row["file_path"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    language=row["language"],
                    score=round(-float(row["score"]), 6),
                    content=row["content"],
                )
            )
    return results