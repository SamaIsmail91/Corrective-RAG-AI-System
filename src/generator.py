"""
Answer generation — the final step, and only run over context that has
already passed relevance grading. The system prompt is intentionally
strict about not using outside knowledge, so hallucination is constrained
both by *what* context reaches the model and by *how* it's told to use it.
"""

from __future__ import annotations

from typing import Dict, List

from src.ollama_client import OllamaClient


GENERATION_SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the user's question using "
    "ONLY the numbered context passages provided below — never use outside "
    "knowledge, even if you're confident about it. Rules:\n"
    "1. Every factual claim must be traceable to a passage. Cite passages "
    "inline like [1] or [2][3], right after the claim they support.\n"
    "2. If the passages only partially answer the question, answer what you "
    "can and explicitly say what's missing.\n"
    "3. If the passages don't answer the question at all, say so plainly — "
    "do not guess or fill gaps with general knowledge.\n"
    "4. Do not invent sources, numbers, names, or citations that are not in "
    "the passages."
)


def _format_context(chunks: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        source = meta.get("source", "unknown source")
        page = meta.get("page")
        loc = f", page {page}" if page else ""
        blocks.append(f"[{i}] (Source: {source}{loc})\n{c['text']}")
    return "\n\n".join(blocks)


class AnswerGenerator:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    def generate(self, question: str, context_chunks: List[Dict]) -> str:
        context_block = _format_context(context_chunks)
        user_prompt = (
            f"Context passages:\n\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the passages above, with inline [n] citations."
        )
        return self.client.chat(
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=self.model,
            temperature=0.2,
        ).strip()
