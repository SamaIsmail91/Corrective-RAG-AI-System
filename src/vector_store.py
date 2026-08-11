"""
Local vector store on top of ChromaDB.

Chroma normally wants to own embedding generation via its own default model
(sentence-transformers, downloaded from the internet). We don't want that
here — every embedding must come from the user's local Ollama server — so
this wrapper always passes embeddings in explicitly, both when adding and
when querying, and never lets Chroma compute one on its own.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import chromadb

from src.document_loader import Chunk


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return
        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "chunk_index": c.chunk_index,
                "page": c.page if c.page is not None else 0,
                "doc_id": c.doc_id,
            }
            for c in chunks
        ]
        self.collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

    def similarity_search(
        self, query_embedding: List[float], k: int = 5, where: Optional[dict] = None
    ) -> List[Dict]:
        if self.collection.count() == 0:
            return []
        k = min(k, self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: List[Dict] = []
        ids = results.get("ids", [[]])[0]
        for i in range(len(ids)):
            distance = results["distances"][0][i]
            out.append(
                {
                    "id": ids[i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": distance,
                    # cosine distance -> a rough, bounded "similarity" for display
                    "similarity": max(0.0, 1.0 - distance / 2.0),
                }
            )
        return out

    # ------------------------------------------------------------------ #
    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> Dict[str, int]:
        if self.collection.count() == 0:
            return {}
        data = self.collection.get(include=["metadatas"])
        counts: Dict[str, int] = {}
        for m in data["metadatas"]:
            src = m.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
