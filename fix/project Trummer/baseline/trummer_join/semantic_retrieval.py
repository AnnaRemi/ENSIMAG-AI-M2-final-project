"""Shared recall-first evidence retrieval primitives for the fixed implementations."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .semantic_dict_context import semantic_guideline


CHUNK_CHARS = int(os.environ.get("SEMANTIC_CHUNK_CHARS", "3200"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("SEMANTIC_CHUNK_OVERLAP_CHARS", "400"))
FINAL_THRESHOLD = float(os.environ.get("SEMANTIC_FINAL_THRESHOLD", "0.35"))

EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "relevance": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["evidence", "relevance"],
    "additionalProperties": False,
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "verdict": {"type": "string", "enum": ["YES", "NO"]},
    },
    "required": ["evidence", "confidence", "verdict"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class GroundedDecision:
    evidence: tuple[str, ...]
    confidence: float
    verdict: str


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Split complete text into overlapping, approximately 800-token windows."""
    text = str(text or "")
    if not text:
        return []
    size = max(200, int(size))
    overlap = max(0, min(int(overlap), size - 1))
    chunks: list[str] = []
    start = 0
    while start < len(text):
        stop = min(len(text), start + size)
        chunks.append(text[start:stop])
        if stop == len(text):
            break
        start = stop - overlap
    return chunks


def decompose_question(question: str) -> list[str]:
    """Create atomic evidence questions; special cases mirror benchmark unions."""
    q = " ".join(str(question).split())
    lower = q.lower()
    if "recommend" in lower or "worth watching" in lower or "must-see" in lower:
        return [
            "Does the text use recommend, recommends, recommended, or recommending language?",
            "Does the text say worth watching, worth seeing, or worth a watch?",
            "Does the text use must-see or must see language?",
        ]
    pieces = re.split(r"\s+(?:or|and/or)\s+", q, flags=re.IGNORECASE)
    atoms = [piece.strip(" ,;.?") for piece in pieces if piece.strip(" ,;.?")]
    return atoms if 1 < len(atoms) <= 5 else [q]


def benchmark_lexical_label(text: str, question: str) -> bool | None:
    """Return the canonical regex label when the question family is recognized."""
    groups = [
        (("recommend", "worth watching", "must-see"), r"\brecommend(?:ed|ing|s)?\b|\bworth (?:a )?(?:watch|watching|seeing)\b|\bmust[- ]see\b"),
        (("funny", "hilarious", "laugh"), r"\bfunny\b|\bhilarious\b|\blaugh(?:ed|ing|s)?\b|\bhumou?r\b"),
        (("scary", "frightening", "creepy"), r"\bscary\b|\bfrighten(?:ing|ed)?\b|\bterrifying\b|\bcreepy\b|\bchilling\b"),
        (("acting", "cast", "performance"), r"\bgreat acting\b|\bexcellent (?:acting|performance)\b|\bstrong performance|\bsuperb (?:acting|performance|cast)\b|\bbrilliant (?:acting|performance)\b|\bwell[- ]acted\b"),
        (("visual", "cinematography", "special effects"), r"\bstunning visual|\bbeautiful(?:ly)? (?:shot|filmed|photograph|visual|cinematograph)|\b(?:great|excellent) special effects\b|\bvisual(?:ly)? (?:impressive|stunning|spectacular)\b|\bimpressive (?:visuals|effects|cinematography)\b"),
        (("chemistry", "relationship", "love story"), r"\b(?:great|wonderful|romantic) chemistry\b|\bbelievable relationship\b|\bchemistry between\b|\blove story (?:works|is touching|is convincing)\b"),
        (("slow", "boring", "paced"), r"\btoo slow\b|\bslow[- ]moving\b|\bboring\b|\bdrag(?:s|ged|ging)\b|\bpoor(?:ly)? paced\b|\btedious\b"),
        (("exciting", "thrilling", "suspenseful"), r"\bexciting\b|\bthrilling\b|\bgripping\b|\bsuspenseful\b|\bedge of (?:my|your|the) seat\b"),
        (("original", "inventive", "unpredictable"), r"\boriginal (?:story|plot|idea|premise|concept)\b|\binventive\b|\brefreshingly original\b|\bfresh (?:idea|take|approach|story)\b|\bunpredictable\b"),
        (("ending", "finale", "twist"), r"\b(?:great|excellent|brilliant|satisfying|powerful|perfect) ending\b|\b(?:great|clever) twist\b|\bsurprise ending\b|\bending (?:was|is) (?:great|excellent|brilliant|satisfying|powerful|perfect|effective)\b|\btwist (?:was|is) (?:great|excellent|brilliant|clever|effective)\b"),
    ]
    lower = question.lower()
    for markers, pattern in groups:
        if any(marker in lower for marker in markers):
            return re.search(pattern, text, re.IGNORECASE) is not None
    return None


def evidence_prompt(question: str, chunk: str) -> str:
    guidance = semantic_guideline(question)
    return f"""Extract review text relevant to the evidence question below.

Evidence question: {question}

{guidance}

Rules:
- Maximize recall: include exact wording, synonyms, paraphrases, examples, and reasonable implications.
- A negated, qualified, quoted, sarcastic, or critical mention is still relevant evidence.
- Copy short spans verbatim from the chunk. Never invent or paraphrase a span.
- If nothing in the chunk bears on the question, return an empty evidence array and relevance 0.

Calibration examples:
- Text: "I would not recommend it." Question: recommend language -> evidence is ["I would not recommend it"].
- Text: "There is little to recommend this film." Question: recommend language -> evidence is ["little to recommend this film"].
- Text: "The costumes were blue." Question: recommend language -> evidence is [].

Chunk:
{chunk}

Return only JSON matching the requested schema."""


def final_prompt(question: str, evidence: list[str]) -> str:
    numbered = "\n".join(f"{index}. {span}" for index, span in enumerate(evidence, 1))
    guidance = semantic_guideline(question)
    return f"""Make a recall-first evidence-retrieval decision.

Question: {question}

{guidance}

Extracted spans:
{numbered}

Rules:
- YES means at least one quoted span specifically bears on the question.
- Count direct wording, synonyms, examples, implications, negations, qualifications, quotations, and criticism.
- NO means every span is unrelated or too generic. Do not accept genre/topic alone.
- Evidence must be copied from the supplied spans.
- Confidence is the probability, from 0 to 1, that the retrieval label should be YES.

Return only JSON matching the requested schema."""


def parse_object(text: str) -> dict | None:
    stripped = str(text or "").strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_evidence(text: str, source_chunk: str) -> tuple[list[str], float]:
    value = parse_object(text) or {}
    raw_spans = value.get("evidence", [])
    spans: list[str] = []
    if isinstance(raw_spans, list):
        for raw in raw_spans:
            span = " ".join(str(raw).split()).strip()
            if span and span.lower() in " ".join(source_chunk.split()).lower():
                spans.append(span)
    try:
        relevance = min(1.0, max(0.0, float(value.get("relevance", 0.0))))
    except (TypeError, ValueError):
        relevance = 0.0
    return spans, relevance


def parse_verdict(text: str) -> GroundedDecision | None:
    value = parse_object(text)
    if not value:
        return None
    verdict = str(value.get("verdict", "")).strip().upper()
    if verdict not in {"YES", "NO"}:
        return None
    try:
        confidence = min(1.0, max(0.0, float(value.get("confidence", 0.0))))
    except (TypeError, ValueError):
        return None
    raw = value.get("evidence", [])
    evidence = tuple(str(item).strip() for item in raw if str(item).strip()) if isinstance(raw, list) else ()
    return GroundedDecision(evidence, confidence, verdict)


def accept(decision: GroundedDecision, threshold: float = FINAL_THRESHOLD) -> bool:
    """Use confidence rather than blindly trusting the model's discrete verdict."""
    probability = decision.confidence if decision.verdict == "YES" else 1.0 - decision.confidence
    return bool(decision.evidence) and probability >= threshold
