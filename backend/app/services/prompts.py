"""Prompt construction for evidence-grounded Q&A.

Pure functions — no I/O. They turn a question plus a set of retrieved
evidence blocks into the message pair sent to the LLM, and they encode the
grounding contract: the model may only answer from the evidence it is given,
must cite evidence by ID, must say when evidence is insufficient, and must
distinguish observed facts from interpretations.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM_INSTRUCTION = (
    "You are an evidence-grounded code analyst for the repository shown in the "
    "evidence blocks below. Your job is to answer the user's question about "
    "that repository, and ONLY about that repository.\n"
    "\n"
    "Grounding rules (non-negotiable):\n"
    "1. Answer ONLY from the evidence blocks supplied below. Never invent, "
    "guess, or recall files, functions, classes, dependencies, architecture, "
    "behaviors, or performance characteristics that are not present in the "
    "evidence.\n"
    "2. Every factual claim must cite the evidence block(s) that support it, "
    "using their IDs in the form [E1], [E2], etc. Do not cite an evidence "
    "block for a claim it does not actually support.\n"
    "3. If the evidence does not contain enough to answer the question, say "
    "explicitly that the evidence is insufficient and set evidence_sufficient "
    "to false. Never stretch or embroider to look helpful.\n"
    "4. Distinguish observed facts (things directly present in the evidence, "
    "e.g. 'the module defines function X at [E2]') from interpretations or "
    "inferences (e.g. 'this may scale to many users') and label them as such.\n"
    "5. Cite only evidence IDs that appear in the supplied blocks. An ID like "
    "[E9] when only E1..E5 exist is forbidden.\n"
    "6. README/documentation evidence is a *claim*, not proof of "
    "implementation. When you cite it, mark it as such and prefer source-code "
    "evidence for claims about how the code behaves.\n"
    "\n"
    "You must respond with ONLY a single JSON object, no prose around it, in "
    "this exact shape:\n"
    '{"answer": "<plain-text answer with [E#] citations inline>", '
    '"citations": ["E1", "E2"], "confidence": "high|medium|low", '
    '"evidence_sufficient": true|false}\n'
    "Where citations is the list of every distinct evidence ID you cited.\n"
)

JSON_INSTRUCTIONS = (
    "Respond with only a single JSON object with exactly these fields:\n"
    '{"answer": "...", "citations": ["E1", ...], '
    '"confidence": "high|medium|low", "evidence_sufficient": true|false}\n'
    "In \"answer\", cite evidence by ID like [E1]. \"citations\" lists every "
    "distinct evidence ID you cited. Set evidence_sufficient=false and say it "
    "plainly if the evidence cannot answer the question."
)


@dataclass(frozen=True)
class EvidenceBlock:
    """One labeled evidence chunk handed to the LLM."""

    id: str  # "E1", "E2", …
    repository: str  # "owner/repo"
    file_path: str
    start_line: int
    end_line: int
    language: str | None
    content: str

    @property
    def formatted(self) -> str:
        lines = [
            f"[{self.id}]",
            f"Repository: {self.repository}",
            f"File: {self.file_path}",
            f"Lines: {self.start_line}-{self.end_line}",
            f"Language: {self.language or 'unknown'}",
            "",
            self.content,
        ]
        return "\n".join(lines)


def format_evidence_block(block: EvidenceBlock) -> str:
    """Render one evidence block as the [E#]-labelled text the spec requires."""
    return block.formatted


def build_qa_messages(
    system_prompt: str,
    blocks: list[EvidenceBlock],
    question: str,
    *,
    json_instructions: str = JSON_INSTRUCTIONS,
) -> list[dict[str, str]]:
    """Assemble the (system, user) message pair for Ollama's /api/chat."""
    evidence_text = "\n\n".join(b.formatted for b in blocks)
    user_content = (
        "Here is the retrieved evidence from the repository:\n\n"
        f"{evidence_text}\n\n"
        f"Question: {question}\n\n"
        f"{json_instructions}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]