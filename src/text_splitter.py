"""
A small, dependency-free recursive-character text splitter.

We intentionally avoid pulling in LangChain just for this: the algorithm is
short and well understood, and keeping it in-house means the whole ingestion
path (extract -> clean -> split) has no hidden network calls or heavyweight
imports, which matters for a "runs entirely offline against Ollama" project.

The approach mirrors the well-known recursive-character strategy: try to
split on the largest, most "natural" separator first (paragraph breaks),
and only fall back to smaller separators (sentences, words, characters) for
pieces that are still too big.
"""

from __future__ import annotations

from typing import List, Sequence


DEFAULT_SEPARATORS: Sequence[str] = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: Sequence[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 5)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators) if separators else list(DEFAULT_SEPARATORS)

    # ------------------------------------------------------------------ #
    def split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        raw_chunks = self._split(text, self.separators)
        raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
        return self._add_overlap(raw_chunks)

    # ------------------------------------------------------------------ #
    def _split(self, text: str, separators: Sequence[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Last resort: hard character slicing.
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        sep, rest = separators[0], separators[1:]
        pieces = text.split(sep) if sep else list(text)

        chunks: List[str] = []
        current = ""
        for i, piece in enumerate(pieces):
            piece_with_sep = piece + (sep if i < len(pieces) - 1 else "")

            if len(current) + len(piece_with_sep) <= self.chunk_size:
                current += piece_with_sep
                continue

            if current:
                chunks.append(current)
                current = ""

            if len(piece_with_sep) > self.chunk_size:
                chunks.extend(self._split(piece_with_sep, rest))
            else:
                current = piece_with_sep

        if current:
            chunks.append(current)
        return chunks

    # ------------------------------------------------------------------ #
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.chunk_overlap :]
            merged = f"{tail} {chunks[i]}" if tail else chunks[i]
            # Don't let overlap blow past ~1.3x chunk size.
            if len(merged) > int(self.chunk_size * 1.3):
                merged = merged[-int(self.chunk_size * 1.3) :]
            overlapped.append(merged)
        return overlapped
