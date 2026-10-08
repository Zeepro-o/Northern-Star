"""Deterministic line-based evidence chunking.

M2 splits indexed source/config/documentation files into chunks that each
carry exact file + line-range provenance. No AST awareness, no embeddings,
no LLM — pure line segmentation that is stable and fully unit-testable.

Design rules
------------
* Small files (<= ``max_lines``) become a single chunk covering the whole file;
  the chunk content is the verbatim file text.
* Large files are split into *sequential* non-overlapping chunks of
  ``chunk_lines`` lines each (the final chunk may be shorter).
* Chunks never split a line; each chunk's content is the "\n"-joined slice.
* Blank files produce no chunks.

Line numbering: lines are 1-indexed. A trailing newline terminates the final
line and does not create an extra (empty) line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Chunk:
    """One evidence-carrying segment of one repository file."""

    repo_id: str
    file_path: str
    language: Optional[str]
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    content: str

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


def segment_lines(
    content: str,
    *,
    chunk_lines: int = 100,
    max_lines: int = 150,
) -> list[tuple[int, int, str]]:
    """Split text into ``(start_line, end_line, content)`` ranges.

    Returns an empty list for blank input.
    """
    if not content:
        return []
    lines = content.splitlines()
    total = len(lines)
    if total <= max_lines:
        return [(1, total, content)]

    chunks: list[tuple[int, int, str]] = []
    for start in range(1, total + 1, chunk_lines):
        end = min(start + chunk_lines - 1, total)
        chunks.append((start, end, "\n".join(lines[start - 1 : end])))
    return chunks


def chunk_file(
    repo_id: str,
    file_path: str,
    content: str,
    language: Optional[str],
    *,
    chunk_lines: int = 100,
    max_lines: int = 150,
) -> list[Chunk]:
    """Chunk one file's content into provenance-carrying ``Chunk`` objects."""
    return [
        Chunk(
            repo_id=repo_id,
            file_path=file_path,
            language=language,
            start_line=start,
            end_line=end,
            content=text,
        )
        for start, end, text in segment_lines(
            content, chunk_lines=chunk_lines, max_lines=max_lines
        )
    ]