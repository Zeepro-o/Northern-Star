"""Evidence-grounded Q&A orchestrator.

Reuses the M2 retrieval layer directly (``search_evidence`` from retrieval.py) —
this module does not duplicate or reimplement retrieval. The flow is:

  question
    → retrieve top_k evidence chunks (BM25, repo-isolated)
    → label as [E1]..[En] blocks with file:line provenance
    → build the grounded prompt
    → call Ollama
    → parse the model output (structured JSON with a plain-text fallback)
    → validate citations against the supplied blocks (invalid IDs rejected)
    → return AnswerResponse (answer, citations, confidence, evidence_sufficient)

When retrieval returns no results, the LLM is never called — the answer is an
immediate "evidence insufficient" response.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from ..config import Settings
from ..models.schemas import AnswerResponse, Citation
from .indexing import evidence_db_path
from .llm import OllamaClient
from .prompts import SYSTEM_INSTRUCTION, EvidenceBlock, build_qa_messages
from .retrieval import search_evidence

# ---------------------------------------------------------------------------
# ParsedAnswer — internal, after parsing the LLM's JSON (or fallback).
# ---------------------------------------------------------------------------


@dataclass
class ParsedAnswer:
    answer: str
    citations: list[str]  # raw "E1" ids, still need validation
    confidence: str  # normalized to high|medium|low
    evidence_sufficient: bool
    raw_was_json: bool


_CONF_RE = re.compile(r"\b(high|medium|low)\b", re.IGNORECASE)
_CITE_ID_RE = re.compile(r"\[E(\d+)\]")  # "[E1]" in answer text
_BARE_ID_RE = re.compile(r"\bE(\d+)\b", re.IGNORECASE)  # "E1" in citations list
_CITE_MARKER_RE = re.compile(r"\[E(\d+)\]")  # for sanitization


# ---------------------------------------------------------------------------
# Citation helpers
# ---------------------------------------------------------------------------


def label_evidence(
    repo_id: str,
    evidence: list,  # list[SearchResult]
) -> list[EvidenceBlock]:
    """Number every retrieved chunk E1..En and return the labelled blocks."""
    blocks: list[EvidenceBlock] = []
    for i, r in enumerate(evidence, start=1):
        blocks.append(
            EvidenceBlock(
                id=f"E{i}",
                repository=repo_id,
                file_path=r.file_path,
                start_line=r.start_line,
                end_line=r.end_line,
                language=r.language,
                content=r.content,
            )
        )
    return blocks


def validate_citations(cited: list[str], valid_ids: set[str]) -> list[str]:
    """Keep only IDs that appear in ``valid_ids``; dedupe while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for c in cited:
        if not isinstance(c, str):
            continue
        raw = c.strip()
        # bare "E1" passes through; "[E1]" is normalized to "E1".
        m = _CITE_ID_RE.match(raw)
        candidate = f"E{m.group(1)}" if m else raw
        if not _BARE_ID_RE.fullmatch(candidate):
            continue  # not an evidence id at all
        candidate = candidate.upper()
        if candidate not in valid_ids or candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def _inline_citation_ids(text: str, valid_ids: set[str]) -> list[str]:
    """Distinct ``[E#]`` markers actually written into the answer text.

    Only IDs that exist among the supplied blocks count. This is the
    *self-anchoring* half of citation validation: whatever the model's
    ``citations`` array declares, only markers the answer text really contains
    can survive to the response.
    """
    out: list[str] = []
    seen: set[str] = set()
    for m in _CITE_ID_RE.finditer(text):
        cid = f"E{m.group(1)}"
        if cid in valid_ids and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def sanitize_answer(text: str, allowed_ids: set[str]) -> str:
    """Remove ``[E#]`` markers whose ID is not in ``allowed_ids``.

    The central invariant: the model cannot make evidence appear. Callers pass
    the set of citations that survived validation, so this also drops inline
    markers the model wrote but never declared (e.g. "[E4]" in the text while
    it declared E1) — and always strips fabricated IDs like [E99].
    """
    return _CITE_MARKER_RE.sub(
        lambda m: m.group(0) if f"E{m.group(1)}" in allowed_ids else "",
        text,
    )


# ---------------------------------------------------------------------------
# LLM output parsing (robust — JSON primary, plain-text fallback).
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> Optional[dict]:
    """Find and parse the first balanced {...} JSON object in text.

    Strips surrounding Markdown code fences first so "```json {…} ```" wrappers
    from models that emit fencing by default do not break parsing.
    """
    s = text.strip()
    # Strip code fences.
    if s.startswith("```"):
        s = re.sub(r"^```\s*\w*\n?", "", s, flags=re.MULTILINE)
        s = re.sub(r"```\s*$", "", s.strip(), flags=re.MULTILINE)
    # Find the first balanced JSON object via bracket counting.
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(s)):
        ch = s[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _normalize_confidence(value: object | None, fallback: str = "medium") -> str:
    if isinstance(value, str) and value.lower().strip() in ("high", "medium", "low"):
        return value.lower().strip()
    return fallback


def _coerce_citations(raw: object | None) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        s = item.strip()
        # Accept "E1" or "[E1]"; normalize to bare "E1".
        m = _CITE_ID_RE.match(s)
        if m:
            s = f"E{m.group(1)}"
        if _BARE_ID_RE.fullmatch(s):
            out.append(s.upper())
    return out


def parse_answer(raw: str, valid_ids: set[str]) -> ParsedAnswer:
    """Parse the LLM's text into a ParsedAnswer.

    Primary: try to extract a JSON object; on failure, fall back to treating the
    whole text as the answer and mining [E#] markers from it.
    ``valid_ids`` is only consulted to warn downstream validation — the parsed
    list may still contain invalid IDs (they are dropped later).
    """
    parsed = _extract_json_object(raw or "")
    if parsed is not None:
        answer = parsed.get("answer", "") if isinstance(parsed, dict) else ""
        if not isinstance(answer, str):
            answer = str(answer)
        citations = _coerce_citations(parsed.get("citations"))
        confidence = _normalize_confidence(parsed.get("confidence"))
        sufficient = parsed.get("evidence_sufficient")
        if not isinstance(sufficient, bool):
            # Missing or null → infer from whether anything was actually said.
            sufficient = bool(citations) and len(answer.strip()) > 0
        evidence_sufficient = bool(sufficient)
        return ParsedAnswer(
            answer=answer,
            citations=citations,
            confidence=confidence,
            evidence_sufficient=evidence_sufficient,
            raw_was_json=True,
        )
    # Fallback — whole text is the answer; mine [E#] markers, conservative.
    text = (raw or "").strip()
    mined = [f"E{m.group(1)}" for m in _CITE_MARKER_RE.finditer(text)]
    conf_match = _CONF_RE.search(text.lower())
    conf = conf_match.group(1).lower() if conf_match else "medium"
    return ParsedAnswer(
        answer=text,
        citations=mined,
        confidence=conf,
        evidence_sufficient=bool(mined),
        raw_was_json=False,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

INSUFFICIENT_ANSWER = (
    "I don't have enough evidence to answer that question confidently. "
    "The retrieved evidence does not contain the information needed to verify "
    "this claim."
)


def answer_question(
    repo_id: str,
    question: str,
    *,
    settings: Settings,
    top_k: Optional[int] = None,
    llm: Optional[OllamaClient] = None,
    retrieve: Callable = search_evidence,  # injectable for tests
) -> AnswerResponse:
    """Answer a question grounded in the repository's evidence index.

    Raises ``RepoNotIndexedError`` when the repo has not been indexed.
    """
    effective_top_k = int(top_k) if top_k is not None else settings.qa_top_k
    db_path = evidence_db_path(settings.storage_root, settings.db_filename)

    # 1 — Retrieve (search_evidence raises RepoNotIndexedError if unindexed).
    evidence = retrieve(
        db_path,
        repo_id,
        question,
        limit=effective_top_k,
        default_limit=effective_top_k,
    )

    # 2 — No results → short-circuit (never call the LLM). This is the only
    # path where confidence/evidence_sufficient are deterministic, so the
    # response says so and claims no evidence grounding.
    if not evidence:
        return AnswerResponse(
            question=question,
            repo_id=repo_id,
            answer=INSUFFICIENT_ANSWER,
            citations=[],
            confidence="low",
            evidence_sufficient=False,
            confidence_source="deterministic",
            evidence_grounding="none",
        )

    # 3 — Label as [E1]..[En] evidence blocks.
    blocks = label_evidence(repo_id, evidence)
    valid_ids = {b.id for b in blocks}

    # 4 — Build the grounded prompt.
    messages = build_qa_messages(SYSTEM_INSTRUCTION, blocks, question)

    # 5 — Call the LLM (Ollama failures propagate to the route layer).
    llm = llm or OllamaClient(
        settings.ollama_base_url,
        settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        think=settings.ollama_think,
    )
    raw = llm.complete(messages)

    # 6 — Parse and validate citations (ID-level, not claim-level).
    # Two independent signals must AGREE for a citation to survive:
    #   declared = the model's "citations" array, checked against real IDs
    #   inline   = the [E#] markers actually present in the answer text
    # An ID the model wrote inline but never declared — or declared but never
    # anchored inline — is dropped, so a self-inconsistent answer (text "[E4]"
    # with array ["E1"]) cannot reach the response carrying provenance it does
    # not actually use. Fabricated IDs like [E99] still never survive.
    parsed = parse_answer(raw, valid_ids)
    declared = validate_citations(parsed.citations, valid_ids)
    inline = _inline_citation_ids(parsed.answer, valid_ids)
    final_ids = [c for c in declared if c in inline]  # declared order kept
    sanitized_answer = sanitize_answer(parsed.answer, set(final_ids))

    # 7 — Resolve Citation objects for every surviving citation (carry file/line
    # provenance). Grounding is derived ONLY from real survivors.
    by_id = {b.id: b for b in blocks}
    citations = [
        Citation(
            id=c_id,
            file_path=by_id[c_id].file_path,
            start_line=by_id[c_id].start_line,
            end_line=by_id[c_id].end_line,
            language=by_id[c_id].language,
            content=by_id[c_id].content,
        )
        for c_id in final_ids
        # Defensive: if somehow an ID slipped through validation, ignore it.
        if c_id in by_id
    ]
    grounding = "cited" if final_ids else "none"

    return AnswerResponse(
        question=question,
        repo_id=repo_id,
        answer=sanitized_answer,
        citations=citations,
        confidence=parsed.confidence,
        evidence_sufficient=parsed.evidence_sufficient,
        confidence_source="model",  # the LLM reported all of the above
        evidence_grounding=grounding,
    )