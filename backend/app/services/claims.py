"""M4 — Claim extraction and evidence verification.

Architecture:
  Claim extraction
    → BM25 evidence retrieval
    → LLM evidence analysis
    → deterministic citation/evidence validation
    → verdict

The LLM may reason about whether evidence supports/refutes a claim, but it
MUST NOT be trusted blindly. The system must validate that every cited
evidence ID actually came from the retrieved evidence.

Verdicts:
  - supported
  - partially_supported
  - unclear
  - contradicted

Critical semantics:
  - No evidence = unclear, NEVER contradicted.
  - Contradicted requires positive evidence that conflicts with the claim.
  - BM25 retrieval alone must NEVER produce "supported".
  - The LLM must only reason over retrieved evidence, never the whole repo.
  - Every explanation must cite exact file:line evidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from ..config import Settings
from ..models.schemas import Claim, SearchResult
from .indexing import evidence_db_path
from .llm import OllamaClient
from .prompts import EvidenceBlock, build_qa_messages
from .qa import (
    INSUFFICIENT_ANSWER,
    _CITE_ID_RE,
    _BARE_ID_RE,
    label_evidence,
    parse_answer,
    sanitize_answer,
    validate_citations,
)
from .retrieval import RepoNotIndexedError, search_evidence


# ---------------------------------------------------------------------------
# System prompt for claim verification
# ---------------------------------------------------------------------------

CLAIM_VERIFICATION_INSTRUCTION = (
    "You are an evidence-grounded claim verifier for the repository shown in the "
    "evidence blocks below. Your job is to evaluate whether the given CLAIM is "
    "supported by the evidence, and ONLY the evidence.\n"
    "\n"
    "Verification rules (non-negotiable):\n"
    "1. Evaluate the CLAIM using ONLY the evidence blocks supplied below. Never "
    "invent, guess, or recall facts not present in the evidence.\n"
    "2. You must cite evidence blocks by their [E#] IDs for every assessment you "
    "make. Do not cite an evidence block for a point it does not actually "
    "support.\n"
    "3. If the evidence does not contain enough information to evaluate the "
    "claim, say explicitly that the evidence is insufficient and set verdict to "
    "\"unclear\". NEVER default to \"contradicted\" when evidence is missing.\n"
    "4. A verdict of \"contradicted\" requires POSITIVE evidence that directly "
    "conflicts with the claim (e.g., claim says \"uses Redis\" but evidence says "
    "\"we use MongoDB\"). Absence of supporting evidence is NOT contradiction.\n"
    "5. \"partially_supported\" means some aspects of the claim are supported by "
    "evidence while others are not supported or are contradicted.\n"
    "6. Cite only evidence IDs that appear in the supplied blocks. An ID like "
    "[E9] when only E1..E5 exist is forbidden.\n"
    "7. README/documentation evidence is a *claim*, not proof of "
    "implementation. When you cite it, mark it as such and prefer source-code "
    "evidence for claims about how the code behaves.\n"
    "\n"
    "You must respond with ONLY a single JSON object, no prose around it, in "
    "this exact shape:\n"
    '{"verdict": "supported|partially_supported|unclear|contradicted", '
    '"explanation": "<plain text with [E#] citations inline>", '
    '"citations": ["E1", "E2"]}\n'
    "Where citations lists every distinct evidence ID you cited.\n"
)

CLAIM_VERIFICATION_JSON_INSTRUCTIONS = (
    "Respond with only a single JSON object with exactly these fields:\n"
    '{"verdict": "supported|partially_supported|unclear|contradicted", '
    '"explanation": "...", "citations": ["E1", ...]}\n'
    "In \"explanation\", cite evidence by ID like [E1]. \"citations\" lists "
    "every distinct evidence ID you cited. Set verdict=\"unclear\" and say it "
    "plainly if the evidence cannot answer the claim."
)


# ---------------------------------------------------------------------------
# Internal parsed answer for claim verification
# ---------------------------------------------------------------------------

@dataclass
class ParsedVerification:
    verdict: str  # supported | partially_supported | unclear | contradicted
    explanation: str
    citations: list[str]  # raw "E1" ids, still need validation
    raw_was_json: bool


# ---------------------------------------------------------------------------
# Parsing and validation (reuse M3 patterns)
# ---------------------------------------------------------------------------

_VERDICT_RE = re.compile(r"\b(supported|partially_supported|unclear|contradicted)\b", re.IGNORECASE)


def _extract_json_object(text: str) -> Optional[dict]:
    """Find and parse the first balanced {...} JSON object in text."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```\s*\w*\n?", "", s, flags=re.MULTILINE)
        s = re.sub(r"```\s*$", "", s.strip(), flags=re.MULTILINE)
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


def _normalize_verdict(value: object | None, fallback: str = "unclear") -> str:
    if isinstance(value, str):
        v = value.lower().strip()
        if v in ("supported", "partially_supported", "unclear", "contradicted"):
            return v
    return fallback


def _coerce_citations(raw: object | None) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        s = item.strip()
        m = _CITE_ID_RE.match(s)
        if m:
            s = f"E{m.group(1)}"
        if _BARE_ID_RE.fullmatch(s):
            out.append(s.upper())
    return out


def parse_verification(raw: str, valid_ids: set[str]) -> ParsedVerification:
    """Parse the LLM's text into a ParsedVerification."""
    parsed = _extract_json_object(raw or "")
    if parsed is not None:
        verdict = _normalize_verdict(parsed.get("verdict"))
        explanation = parsed.get("explanation", "") if isinstance(parsed, dict) else ""
        if not isinstance(explanation, str):
            explanation = str(explanation)
        citations = _coerce_citations(parsed.get("citations"))
        return ParsedVerification(
            verdict=verdict,
            explanation=explanation,
            citations=citations,
            raw_was_json=True,
        )
    # Fallback — whole text is the explanation; mine [E#] markers
    text = (raw or "").strip()
    mined = [f"E{m.group(1)}" for m in _CITE_ID_RE.finditer(text)]
    verdict_match = _VERDICT_RE.search(text.lower())
    verdict = verdict_match.group(1).lower() if verdict_match else "unclear"
    return ParsedVerification(
        verdict=verdict,
        explanation=text,
        citations=mined,
        raw_was_json=False,
    )


# ---------------------------------------------------------------------------
# Public entry point: verify a single claim
# ---------------------------------------------------------------------------

UNSUFFICIENT_VERIFICATION = (
    "I don't have enough evidence to evaluate this claim. "
    "The retrieved evidence does not contain the information needed to verify "
    "or refute it."
)


def verify_claim(
    repo_id: str,
    claim: Claim,
    *,
    settings: Settings,
    top_k: Optional[int] = None,
    llm: Optional[OllamaClient] = None,
    retrieve: Callable = search_evidence,
) -> Claim:
    """Verify a claim against the repository's evidence index.

    Returns the claim with `verdict`, `verdict_explanation`, and `evidence_ids`
    populated. Raises `RepoNotIndexedError` when the repo has not been indexed.
    """
    effective_top_k = int(top_k) if top_k is not None else settings.qa_top_k
    db_path = evidence_db_path(settings.storage_root, settings.db_filename)

    # 1 — Retrieve evidence using the claim text as the BM25 query
    evidence = retrieve(
        db_path,
        repo_id,
        claim.text,
        limit=effective_top_k,
        default_limit=effective_top_k,
    )

    # 2 — No results → unclear (NEVER contradicted)
    if not evidence:
        claim.verdict = "unclear"
        claim.verdict_explanation = UNSUFFICIENT_VERIFICATION
        claim.evidence_ids = []
        claim.repo_id = repo_id
        return claim

    # 3 — Label as [E1]..[En] evidence blocks
    blocks = label_evidence(repo_id, evidence)
    valid_ids = {b.id for b in blocks}

    # 4 — Build the verification prompt
    claim_text = f"CLAIM: {claim.text}\n\nSOURCE: {claim.source} (kind: {claim.kind.value})"
    messages = build_qa_messages(
        CLAIM_VERIFICATION_INSTRUCTION,
        blocks,
        claim_text,
        json_instructions=CLAIM_VERIFICATION_JSON_INSTRUCTIONS,
    )

    # 5 — Call the LLM
    llm = llm or OllamaClient(
        settings.ollama_base_url,
        settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        think=settings.ollama_think,
    )
    raw = llm.complete(messages)

    # 6 — Parse and validate citations (ID-level, same as M3.1)
    parsed = parse_verification(raw, valid_ids)
    declared = validate_citations(parsed.citations, valid_ids)
    inline = _inline_citation_ids(parsed.explanation, valid_ids)

    # Primary: citations with both declared AND inline markers (strict)
    primary_ids = [c for c in declared if c in inline]

    # Recovery: if primary is empty but LLM declared valid IDs, attempt safe recovery
    recovered_ids: list[str] = []
    if not primary_ids and declared:
        recovered_ids = _recover_citations(
            explanation=parsed.explanation,
            declared=declared,
            valid_ids=valid_ids,
            blocks={b.id: b for b in blocks},
        )

    # Combine: primary first, then recovered (preserving order)
    final_ids = primary_ids + [c for c in recovered_ids if c not in primary_ids]
    sanitized_explanation = sanitize_answer(parsed.explanation, set(final_ids))

    # 7 — Map verdict: if no validated citations, force "unclear"
    if not final_ids:
        verdict = "unclear"
    else:
        verdict = parsed.verdict
        # Safety: no evidence can never produce "supported" or "contradicted"
        if verdict in ("supported", "contradicted", "partially_supported") and not final_ids:
            verdict = "unclear"

    # 8 — Resolve Citation objects for every surviving citation (carry file/line provenance)
    by_id = {b.id: b for b in blocks}
    evidence_ids = [
        c_id for c_id in final_ids if c_id in by_id
    ]

    claim.verdict = verdict
    claim.verdict_explanation = sanitized_explanation
    claim.evidence_ids = evidence_ids
    claim.repo_id = repo_id

    return claim


def _inline_citation_ids(text: str, valid_ids: set[str]) -> list[str]:
    """Distinct ``[E#]`` markers actually written into the text."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _CITE_ID_RE.finditer(text):
        cid = f"E{m.group(1)}"
        if cid in valid_ids and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _recover_citations(
    explanation: str,
    declared: list[str],
    valid_ids: set[str],
    blocks: dict[str, EvidenceBlock],
) -> list[str]:
    """Safely recover citations that were declared but lack inline markers.

    Recovery is allowed ONLY when ALL conditions hold:
    1. The LLM declared the evidence ID in its citations array
    2. The evidence ID was actually supplied to the LLM (in valid_ids)
    3. The evidence ID passes deterministic validation (not fabricated)
    4. The explanation text semantically refers to that evidence (contains
       key terms from the evidence content)

    This prevents hallucinated citations while recovering from the common
    LLM failure mode of declaring citations but forgetting inline markers.
    """
    recovered: list[str] = []

    # Pre-compute evidence content keywords for semantic matching
    evidence_keywords: dict[str, set[str]] = {}
    for eid, block in blocks.items():
        # Extract meaningful tokens from evidence content (identifiers, keywords)
        content_lower = block.content.lower()
        # Get alphanumeric tokens of length >= 4 (stricter)
        tokens = set(re.findall(r"[a-z0-9_]{4,}", content_lower))
        # Filter out very common words
        stopwords = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "has", "was", "one", "our", "out", "get",
            "use", "used", "using", "this", "that", "with", "from", "have", "had", "will", "would", "could", "should",
            "may", "might", "must", "shall", "into", "onto", "upon", "over", "under", "again", "also", "such", "than",
            "then", "when", "where", "which", "while", "after", "before", "since", "until", "unless", "because",
            "through", "during", "without", "within", "between", "among", "about", "above", "below", "beyond",
            "around", "across", "against", "along", "inside", "outside", "throughout", "despite", "except",
            "toward", "towards", "evidence", "shows", "show", "showed", "shown", "states", "state", "stated",
            "claim", "claims", "claimed", "file", "files", "code", "codes", "line", "lines", "function", "functions",
            "class", "classes", "module", "modules", "import", "imports", "from", "def", "return", "returns",
            "true", "false", "none", "null", "test", "tests", "testing", "example", "examples", "simple",
        }
        tokens = {t for t in tokens if t not in stopwords}
        evidence_keywords[eid] = tokens

    # Also extract tokens from explanation
    expl_tokens = set(re.findall(r"[a-z0-9_]{4,}", explanation.lower()))
    expl_tokens = {t for t in expl_tokens if t not in stopwords}

    for cid in declared:
        if cid not in valid_ids:
            continue  # Not supplied to LLM
        if cid not in blocks:
            continue  # Not in evidence blocks

        # Check if explanation semantically refers to this evidence
        # by looking for token overlap
        block_tokens = evidence_keywords.get(cid, set())
        if block_tokens:
            overlap = block_tokens & expl_tokens
            # Require at least 3 meaningful token overlaps for recovery (stricter)
            if len(overlap) >= 3:
                recovered.append(cid)

    return recovered