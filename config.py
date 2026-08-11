"""
Central configuration for the Corrective RAG System.

Nothing in this project talks to a paid, key-gated API. All language-model
and embedding calls go to a locally running Ollama server
(https://ollama.com), so the only "credential" involved is the URL of that
server (which defaults to http://localhost:11434 and needs no key).

Everything here can be overridden from the Streamlit sidebar at runtime;
these are just the defaults a fresh install starts with.
"""

from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Ollama connection
# --------------------------------------------------------------------------- #
OLLAMA_HOST_DEFAULT = "http://localhost:11434"
OLLAMA_REQUEST_TIMEOUT = 180  # seconds, generous for slower local hardware

# Sensible fallback model names if the user hasn't pulled anything custom.
# These are only used to pre-select a dropdown item when the name is present
# in `ollama list` — the app never assumes a model exists without checking.
SUGGESTED_LLM_MODELS = ["llama3.1", "llama3.2", "qwen2.5", "mistral", "phi3"]
SUGGESTED_EMBED_MODELS = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]


# --------------------------------------------------------------------------- #
# Ingestion / chunking
# --------------------------------------------------------------------------- #
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 150        # characters of overlap between consecutive chunks
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
PERSIST_DIR = "vectordb"
COLLECTION_NAME = "corrective_rag_docs"


# --------------------------------------------------------------------------- #
# Corrective RAG behaviour
# --------------------------------------------------------------------------- #
TOP_K = 5                          # chunks pulled per retrieval attempt
MIN_RELEVANT_CHUNKS = 2            # verified chunks needed before we stop correcting
MAX_CORRECTION_ITERATIONS = 2      # extra retrieval attempts after the first
CONFIDENCE_THRESHOLD = 0.5         # grader confidence needed to keep a chunk
MAX_CONTEXT_CHUNKS = 6             # cap on chunks fed to the generator
ENABLE_WEB_SEARCH_DEFAULT = False  # opt-in corrective fallback

UNANSWERED_MESSAGE = (
    "I couldn't find reliable evidence for this in the knowledge base. "
    "I looked, graded the results, rewrote the question, and tried again — "
    "but nothing retrieved cleared the relevance bar, so I'm not going to "
    "guess. Try rephrasing the question, or add a document that covers this "
    "topic."
)


@dataclass
class RuntimeConfig:
    """Mutable, per-session configuration driven by the Streamlit sidebar."""

    ollama_host: str = OLLAMA_HOST_DEFAULT
    llm_model: str = ""
    embed_model: str = ""

    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP

    top_k: int = TOP_K
    min_relevant_chunks: int = MIN_RELEVANT_CHUNKS
    max_iterations: int = MAX_CORRECTION_ITERATIONS
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    max_context_chunks: int = MAX_CONTEXT_CHUNKS
    enable_web_search: bool = ENABLE_WEB_SEARCH_DEFAULT

    persist_dir: str = PERSIST_DIR
    collection_name: str = COLLECTION_NAME
