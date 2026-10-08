"""Tests for deterministic line-based evidence chunking."""

from __future__ import annotations

from app.services.chunking import Chunk, chunk_file, segment_lines


class TestSegmentLines:
    def test_blank_content_has_no_chunks(self):
        assert segment_lines("") == []
        assert segment_lines("\n\n") != []  # whitespace still has lines

    def test_whitespace_only_counts_as_lines(self):
        assert segment_lines("\n\n") == [(1, 2, "\n\n")]

    def test_small_file_is_one_chunk(self):
        content = "a\nb\nc\n"
        assert segment_lines(content, chunk_lines=100, max_lines=150) == [(1, 3, content)]

    def test_exact_boundary_is_single_chunk(self):
        content = "\n".join(str(i) for i in range(150))
        chunks = segment_lines(content, chunk_lines=100, max_lines=150)
        assert chunks == [(1, 150, content)]

    def test_boundary_plus_one_splits(self):
        content = "\n".join(str(i) for i in range(151))
        chunks = segment_lines(content, chunk_lines=100, max_lines=150)
        assert len(chunks) == 2
        assert chunks[0][:2] == (1, 100)
        assert chunks[1][:2] == (101, 151)

    def test_large_file_chunk_offsets_and_contents(self):
        lines = [f"line{i}" for i in range(1, 251)]
        content = "\n".join(lines)
        chunks = segment_lines(content, chunk_lines=100, max_lines=150)
        assert [(s, e) for s, e, _ in chunks] == [(1, 100), (101, 200), (201, 250)]
        assert chunks[0][2] == "\n".join(lines[0:100])
        assert chunks[1][2] == "\n".join(lines[100:200])
        assert chunks[2][2] == "\n".join(lines[200:250])

    def test_chunks_never_split_a_line(self):
        # Chunks only break on line boundaries: concatenating chunk contents
        # (with "\n" separators) reproduces the source exactly.
        content = "\n".join(f"l{i}" for i in range(230))
        pieces = segment_lines(content, chunk_lines=80, max_lines=100)
        joined = "\n".join(text for _, _, text in pieces)
        assert joined == content  # lossless round-trip

    def test_no_trailing_newline_phantom_line(self):
        # "a\nb" = two real lines; "a\nb\n" = two real lines too.
        assert segment_lines("a\nb") == [(1, 2, "a\nb")]
        assert segment_lines("a\nb\n") == [(1, 2, "a\nb\n")]


class TestChunkFile:
    def test_blank_file_no_chunks(self):
        assert chunk_file("acme/ev", "x.py", "", "Python") == []

    def test_small_file_provenance(self):
        chunks = chunk_file(
            "acme/ev", "src/thing.py", "def f():\n    return 1\n", "Python",
            chunk_lines=100, max_lines=150,
        )
        assert len(chunks) == 1
        c = chunks[0]
        assert c.repo_id == "acme/ev"
        assert c.file_path == "src/thing.py"
        assert c.language == "Python"
        assert (c.start_line, c.end_line) == (1, 2)
        assert c.line_count == 2
        assert c.content == "def f():\n    return 1\n"

    def test_chunks_carry_consistent_provenance(self):
        content = "\n".join(f"x{i}" for i in range(300))
        chunks = chunk_file("acme/ev", "big.py", content, "Python",
                            chunk_lines=100, max_lines=150)
        assert len(chunks) == 3
        assert all(c.repo_id == "acme/ev" and c.file_path == "big.py" for c in chunks)
        assert [(c.start_line, c.end_line) for c in chunks] == [
            (1, 100), (101, 200), (201, 300)
        ]
        last = chunks[-1]
        assert last.end_line - last.start_line + 1 == last.content.count("\n") + 1

    def test_final_chunk_may_be_short(self):
        content = "\n".join(f"y{i}" for i in range(103))
        chunks = chunk_file("acme/ev", "shortish.py", content, "Python",
                            chunk_lines=100, max_lines=50)
        assert [(c.start_line, c.end_line) for c in chunks] == [(1, 100), (101, 103)]

    def test_language_is_optional(self):
        chunks = chunk_file("acme/ev", "notes.txt", "hello\n", None)
        assert chunks[0].language is None