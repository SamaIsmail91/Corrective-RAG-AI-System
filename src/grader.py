"""
Relevance grading — the "C" in Corrective RAG.

Every chunk that comes back from the vector store is graded against the
*original* user question by the LLM before it's trusted. A chunk only
survives into the answer-generation step if the grader marks it relevant
AND its confidence clears the configured threshold. This is what lets the
pipeline detect "the retrieval was weak" and trigger a query rewrite +
re-retrieval instead of generating from noise.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List

from src.ollama_client import OllamaClient, OllamaError


GRADE_SYSTEM_PROMPT = (
    "You are a strict relevance grader for a retrieval system. Given a "
    "user question and a single retrieved passage, decide whether the "
    "passage contains information that would help answer the question. "
    "Be strict: a passage that is only loosely or tangentially related is "
    "NOT relevant. Respond with JSON only, no other text, in exactly this "
    'shape: {"relevant": true or false, "confidence": a number from 0 to 1, '
    '"reason": "one short sentence"}.'
)

GRADE_USER_TEMPLATE = "Question: {question}\n\nRetrieved passage:\n{passage}"


@dataclass
class Grade:
    relevant: bool
    confidence: float
    reason: str


class RelevanceGrader:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    # ------------------------------------------------------------------ #
    def grade(self, question: str, passage_text: str) -> Grade:
        prompt = GRADE_USER_TEMPLATE.format(
            question=question, passage=passage_text[:2500]
        )
        raw = ""
        try:
            raw = self.client.chat(
                messages=[
                    {"role": "system", "content": GRADE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                json_mode=True,
                temperature=0.0,
            )
            data = json.loads(raw)
            return Grade(
                relevant=bool(data.get("relevant", False)),
                confidence=float(data.get("confidence", 0.5)),
                reason=str(data.get("reason", "")).strip(),
            )
        except OllamaError:
            raise
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            return self._fallback_parse(raw)

    def grade_batch(self, question: str, chunks: List[Dict]) -> List[Grade]:
        return [self.grade(question, c["text"]) for c in chunks]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _fallback_parse(raw: str) -> Grade:
        """If the model didn't return clean JSON, fall back to a cautious
        keyword read of the raw text rather than failing the whole run."""
        lowered = (raw or "").lower()
        relevant = bool(re.search(r"\btrue\b|\byes\b|\brelevant\b", lowered)) and not (
            re.search(r"\bnot relevant\b|\birrelevant\b|\bfalse\b", lowered)
        )
        return Grade(
            relevant=relevant,
            confidence=0.35 if relevant else 0.15,
            reason="Grader response could not be parsed as JSON; used a cautious fallback read.",
        )
