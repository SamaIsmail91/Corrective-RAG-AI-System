"""
Query rewriting — used when the graded retrieval was weak or empty.

The rewriter is deliberately told *why* the previous attempt was weak
(no relevant passages vs. only partial coverage) so it can push the query
in a useful direction rather than just paraphrasing it.
"""

from __future__ import annotations

from src.ollama_client import OllamaClient


REWRITE_SYSTEM_PROMPT = (
    "You rewrite search queries for a vector-similarity document retriever. "
    "You will be given the user's original question, the search query that "
    "was just tried, and why it failed. Produce ONE improved search query: "
    "more specific, keyword-rich, with abbreviations expanded and ambiguity "
    "removed. Do not answer the question. Return ONLY the rewritten query "
    "text with no quotes, labels, or explanation."
)

REWRITE_USER_TEMPLATE = (
    "Original question: {question}\n"
    "Previous search query: {previous_query}\n"
    "Why it failed: {reason}\n\n"
    "Rewritten search query:"
)


class QueryRewriter:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    def rewrite(self, question: str, previous_query: str, reason: str) -> str:
        prompt = REWRITE_USER_TEMPLATE.format(
            question=question, previous_query=previous_query, reason=reason
        )
        content = self.client.chat(
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=0.4,
        )
        cleaned = content.strip().strip('"').strip("'").strip()
        # Guard against the model returning nothing usable.
        return cleaned if cleaned else question
