"""Tests for the prompt builder: grounding contract and message assembly."""

from __future__ import annotations

from app.services.prompts import (
    SYSTEM_INSTRUCTION,
    JSON_INSTRUCTIONS,
    EvidenceBlock,
    build_qa_messages,
    format_evidence_block,
)


def _block(id: str, path: str = "src/auth/middleware.py", start: int = 12,
           end: int = 48, lang: str = "Python", content: str = "def require_auth(request):\n    pass") -> EvidenceBlock:
    return EvidenceBlock(
        id=id,
        repository="acme/evidence",
        file_path=path,
        start_line=start,
        end_line=end,
        language=lang,
        content=content,
    )


class TestSystemInstruction:
    def test_forbids_inventing_files_and_behavior(self):
        assert "never invent" in SYSTEM_INSTRUCTION.lower()

    def test_restricts_answer_to_supplied_evidence(self):
        assert "ONLY from the evidence" in SYSTEM_INSTRUCTION

    def test_requires_citations(self):
        assert "cite" in SYSTEM_INSTRUCTION.lower()
        assert "[E1]" in SYSTEM_INSTRUCTION

    def test_requires_explicit_insufficient_statement(self):
        assert "insufficient" in SYSTEM_INSTRUCTION.lower()
        assert "evidence_sufficient" in SYSTEM_INSTRUCTION

    def test_distinguishes_facts_and_interpretations(self):
        assert "observed facts" in SYSTEM_INSTRUCTION
        assert "interpretations" in SYSTEM_INSTRUCTION.lower()

    def test_requires_json_only_response(self):
        assert "JSON object" in SYSTEM_INSTRUCTION

    def test_readme_claims_are_marked_not_proof(self):
        assert "claim" in SYSTEM_INSTRUCTION.lower()
        assert "README" in SYSTEM_INSTRUCTION


class TestEvidenceBlockFormatting:
    def test_renders_all_provenance_headers(self):
        block = _block("E1")
        text = format_evidence_block(block)
        assert text.startswith("[E1]\n")
        assert "Repository: acme/evidence" in text
        assert "File: src/auth/middleware.py" in text
        assert "Lines: 12-48" in text
        assert "Language: Python" in text
        assert "def require_auth(request):\n    pass" in text

    def test_unknown_language_rendered(self):
        block = _block("E2", lang=None)
        assert "Language: unknown" in format_evidence_block(block)

    def test_formatted_property_matches_function(self):
        block = _block("E3")
        assert block.formatted == format_evidence_block(block)


class TestBuildMessages:
    def test_messages_have_system_then_user(self):
        msgs = build_qa_messages(SYSTEM_INSTRUCTION, [], "where are sessions?")
        assert [m["role"] for m in msgs] == ["system", "user"]

    def test_question_in_user_message_not_system(self):
        msgs = build_qa_messages(SYSTEM_INSTRUCTION, [], "Where is the vault?")
        assert "Where is the vault?" in msgs[1]["content"]
        assert "Where is the vault?" not in msgs[0]["content"]

    def test_every_evidence_block_included(self):
        blocks = [
            _block("E1", path="src/api/routes.py"),
            _block("E2", path="src/db/connection.py", start=5, end=9),
        ]
        msgs = build_qa_messages(SYSTEM_INSTRUCTION, blocks, "how is JSON handled?")
        user = msgs[1]["content"]
        assert "[E1]" in user and "[E2]" in user
        assert "src/api/routes.py" in user and "src/db/connection.py" in user
        assert "Lines: 5-9" in user
        # Block count informs how many distinct-labelled chunks are rendered.
        assert user.count("[E") >= 2

    def test_evidence_block_count_matches(self):
        blocks = [_block(f"E{i}") for i in range(1, 4)]
        user = build_qa_messages(SYSTEM_INSTRUCTION, blocks, "q")[1]["content"]
        # Blocks render as block-headers "[E1]\nRepository: ..." — one per chunk.
        import re
        headers = [f"E{n}" for n in re.findall(r"^\[E(\d+)\]$", user, re.MULTILINE)]
        assert headers == ["E1", "E2", "E3"]
        # No block beyond the supplied set is ever rendered.
        assert "[E4]" not in user

    def test_user_message_requests_structured_json(self):
        msgs = build_qa_messages(SYSTEM_INSTRUCTION, [_block("E1")], "q")
        assert "JSON object" in msgs[1]["content"]
        assert "evidence_sufficient" in msgs[1]["content"]

    def test_json_instructions_mention_required_fields(self):
        assert '"answer"' in JSON_INSTRUCTIONS
        assert '"citations"' in JSON_INSTRUCTIONS
        assert '"confidence"' in JSON_INSTRUCTIONS
        assert '"evidence_sufficient"' in JSON_INSTRUCTIONS

    def test_content_rendered_verbatim(self):
        block = _block("E1", content="raw_binary_token == 'x'")
        text = format_evidence_block(block)
        assert "raw_binary_token == 'x'" in text