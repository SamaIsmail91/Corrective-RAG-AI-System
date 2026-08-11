"""
Ties together document_loader -> Ollama embeddings -> VectorStore, as a
generator so the Streamlit UI can show a live per-file, per-chunk progress
bar instead of freezing during ingestion (embedding a large document one
chunk at a time against a local model can take a while).
"""

from __future__ import annotations

from typing import Dict, Iterator

from config import RuntimeConfig
from src.document_loader import UnsupportedFileType, load_and_chunk
from src.ollama_client import OllamaClient
from src.vector_store import VectorStore


def ingest_file_stream(
    path: str,
    display_name: str,
    client: OllamaClient,
    cfg: RuntimeConfig,
    vector_store: VectorStore,
) -> Iterator[Dict]:
    yield {"type": "status", "message": f"Extracting and cleaning {display_name}…"}
    try:
        chunks = load_and_chunk(
            path,
            display_name=display_name,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
    except UnsupportedFileType as e:
        yield {"type": "error", "message": str(e)}
        return
    except Exception as e:
        yield {"type": "error", "message": f"Could not read {display_name}: {e}"}
        return

    if not chunks:
        yield {
            "type": "error",
            "message": f"No extractable text found in {display_name} (it may be a scanned/image-only file).",
        }
        return

    yield {"type": "chunked", "count": len(chunks)}

    embeddings = []
    for i, c in enumerate(chunks):
        embeddings.append(client.embed(c.text, cfg.embed_model))
        yield {"type": "embed_progress", "done": i + 1, "total": len(chunks)}

    vector_store.add_chunks(chunks, embeddings)
    yield {"type": "done", "source": display_name, "count": len(chunks)}
