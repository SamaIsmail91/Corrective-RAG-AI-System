"""
Evidence Desk — a Corrective RAG System.

Run with:  streamlit run app.py

Everything here runs against a local Ollama server — no API key required.
"""

from __future__ import annotations
import html as html_lib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from config import (
    RuntimeConfig,
    SUGGESTED_EMBED_MODELS,
    SUGGESTED_LLM_MODELS,
)
from src.grader import Grade
from src.ingestion_service import ingest_file_stream
from src.ollama_client import OllamaClient, OllamaError
from src.pipeline import CorrectiveRAGPipeline
from src.vector_store import VectorStore
from src.web_search import is_available as is_web_search_available

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTORDB_DIR = BASE_DIR / "vectordb"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORDB_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Page setup + styling
# =============================================================================
st.set_page_config(
    page_title="Evidence Desk · Corrective RAG",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700;9..144,900&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink: #1B1F2A;
    --ink-soft: #262C3D;
    --paper: #EFEDE4;
    --paper-card: #FBFAF5;
    --graphite: #5B6270;
    --graphite-light: #9096A1;
    --border: #DBD7C9;
    --verified: #2F8F72;
    --verified-bg: #E3F1EA;
    --corrective: #C98A2C;
    --corrective-bg: #FBF0DE;
    --flagged: #B24B3C;
    --flagged-bg: #FAEAE7;
}

html, body, [class^="css"], .stMarkdown, .stText { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background-color: var(--paper); }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background-color: var(--ink); }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {
    color: var(--paper) !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] p { color: inherit !important; }
[data-testid="stSidebar"] hr { border-color: rgba(239,237,228,0.15); }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background-color: var(--ink-soft) !important;
    border-color: rgba(239,237,228,0.2) !important;
    color: var(--paper) !important;
}

.brand { font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.5rem; letter-spacing: -0.01em; margin-bottom: 0; color: var(--paper); }
.brand-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; letter-spacing: 0.14em; color: var(--graphite-light) !important; margin-bottom: 1.2rem; text-transform: uppercase; }
.sidebar-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--graphite-light) !important; margin: 0.6rem 0 0.3rem 0; }

/* ---------- Status dot ---------- */
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.status-ok { background-color: #4FBF8F; box-shadow: 0 0 6px #4FBF8F; }
.status-bad { background-color: #D9695C; }

/* ---------- Main header ---------- */
.app-title { font-family: 'Fraunces', serif; font-weight: 800; font-size: 2.3rem; color: var(--ink); letter-spacing: -0.02em; margin-bottom: 0.1rem; }
.app-subtitle { font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--graphite); margin-bottom: 0.4rem; }

/* ---------- Stamp badge (signature element) ---------- */
.stamp { display: inline-block; font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.72rem; letter-spacing: 0.07em; padding: 5px 12px; border: 2px solid currentColor; border-radius: 3px; transform: rotate(-1deg); margin-bottom: 0.7rem; }
.stamp-verified { color: var(--verified); background: var(--verified-bg); }
.stamp-flagged { color: var(--flagged); background: var(--flagged-bg); }

/* ---------- Grading pills ---------- */
.pill { display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; padding: 2px 8px; border-radius: 10px; margin-right: 6px; white-space: nowrap; }
.pill-verified { background: var(--verified-bg); color: var(--verified); }
.pill-flagged { background: var(--flagged-bg); color: var(--flagged); }
.mono-note { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--graphite); }

/* ---------- Evidence card ---------- */
.evidence-card { background: var(--paper-card); border: 1px solid var(--border); border-left: 3px solid var(--verified); border-radius: 4px; padding: 10px 14px; margin-bottom: 10px; }
.evidence-head { display: flex; gap: 10px; align-items: baseline; margin-bottom: 4px; flex-wrap: wrap; }
.evidence-tag { font-family: 'IBM Plex Mono', monospace; font-weight: 600; color: var(--verified); }
.evidence-source { font-weight: 600; color: var(--ink); font-size: 0.92rem; }
.evidence-confidence { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--graphite); margin-left: auto; }
.evidence-body { font-size: 0.86rem; color: var(--graphite); line-height: 1.5; }

/* ---------- How-it-works process rows ---------- */
.process-row { display: flex; gap: 14px; align-items: flex-start; padding: 14px 0; border-bottom: 1px solid var(--border); }
.process-num { font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.3rem; color: var(--graphite-light); width: 34px; flex-shrink: 0; }
.process-title { font-weight: 600; color: var(--ink); margin-bottom: 2px; }
.process-body { color: var(--graphite); font-size: 0.9rem; line-height: 1.5; }

.stButton>button { border-radius: 4px; font-weight: 500; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================
# Cached resources
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_client(host: str) -> OllamaClient:
    return OllamaClient(host)


@st.cache_resource(show_spinner=False)
def get_vector_store(persist_dir: str, collection_name: str) -> VectorStore:
    return VectorStore(persist_dir, collection_name)


@st.cache_data(ttl=8, show_spinner=False)
def check_ollama(host: str) -> bool:
    return OllamaClient(host).is_available()


@st.cache_data(ttl=8, show_spinner=False)
def list_models_cached(host: str) -> List[str]:
    return OllamaClient(host).list_models()


def _default_index(options: List[str], preferred: List[str] | str) -> int:
    if not options:
        return 0
    if isinstance(preferred, str) and preferred in options:
        return options.index(preferred)
    if isinstance(preferred, list):
        for p in preferred:
            for i, opt in enumerate(options):
                if p in opt:
                    return i
    return 0


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir=str(UPLOAD_DIR)
    ) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


# =============================================================================
# Session state
# =============================================================================
if "cfg" not in st.session_state:
    cfg = RuntimeConfig()
    cfg.persist_dir = str(VECTORDB_DIR)
    st.session_state.cfg = cfg
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict] = []
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

cfg: RuntimeConfig = st.session_state.cfg


# =============================================================================
# Sidebar — control panel
# =============================================================================
with st.sidebar:
    st.markdown('<div class="brand">🗂️ Evidence Desk</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Corrective RAG · Control Panel</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Ollama connection</div>', unsafe_allow_html=True)
    cfg.ollama_host = st.text_input(
        "Server URL", value=cfg.ollama_host, label_visibility="collapsed"
    )
    available = check_ollama(cfg.ollama_host)
    client = get_client(cfg.ollama_host)

    status_col, refresh_col = st.columns([4, 1])
    with status_col:
        dot = "status-ok" if available else "status-bad"
        label = "Connected" if available else "Not reachable"
        st.markdown(f'<span class="status-dot {dot}"></span>{label}', unsafe_allow_html=True)
    with refresh_col:
        if st.button("↻", help="Re-check connection"):
            check_ollama.clear()
            list_models_cached.clear()
            st.rerun()

    if not available:
        st.caption("Start Ollama (`ollama serve`) and confirm the URL above.")

    models = list_models_cached(cfg.ollama_host) if available else []

    st.markdown('<div class="sidebar-label">Models</div>', unsafe_allow_html=True)
    if not models:
        st.warning("No local models detected yet.")
        st.code("ollama pull llama3.1\nollama pull nomic-embed-text", language="bash")

    llm_options = models or ["— none available —"]
    embed_options = models or ["— none available —"]
    cfg.llm_model = st.selectbox(
        "Language model — grading, rewriting & answers",
        options=llm_options,
        index=_default_index(llm_options, cfg.llm_model or SUGGESTED_LLM_MODELS),
    )
    cfg.embed_model = st.selectbox(
        "Embedding model",
        options=embed_options,
        index=_default_index(embed_options, cfg.embed_model or SUGGESTED_EMBED_MODELS),
    )
    if not models:
        cfg.llm_model, cfg.embed_model = "", ""

    st.divider()
    st.markdown('<div class="sidebar-label">Upload documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT, or MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    process_clicked = st.button(
        "Process documents",
        disabled=not uploaded_files or not available or not cfg.embed_model,
        use_container_width=True,
    )

    vector_store = get_vector_store(cfg.persist_dir, cfg.collection_name)

    if process_clicked and uploaded_files:
        for f in uploaded_files:
            tmp_path = save_uploaded_file(f)
            progress = st.progress(0.0, text=f"Processing {f.name}…")
            try:
                for event in ingest_file_stream(tmp_path, f.name, client, cfg, vector_store):
                    if event["type"] == "embed_progress":
                        frac = event["done"] / max(event["total"], 1)
                        progress.progress(
                            frac, text=f"Embedding {f.name}: {event['done']}/{event['total']} chunks"
                        )
                    elif event["type"] == "error":
                        st.error(event["message"])
                    elif event["type"] == "done":
                        st.success(f"✓ {event['source']} — {event['count']} chunks indexed.")
            except OllamaError as e:
                st.error(f"Ingestion stopped: {e}")
            finally:
                progress.empty()
                Path(tmp_path).unlink(missing_ok=True)
        st.rerun()

    st.divider()
    st.markdown('<div class="sidebar-label">Knowledge base</div>', unsafe_allow_html=True)
    sources = vector_store.list_sources()
    if sources:
        for src, count in sorted(sources.items()):
            c1, c2 = st.columns([4, 1])
            c1.caption(f"📄 {src} · {count} chunks")
            if c2.button("✕", key=f"del_{src}", help=f"Remove {src}"):
                vector_store.delete_source(src)
                st.rerun()

        if st.session_state.confirm_clear:
            st.warning("Clear the entire knowledge base? This can't be undone.")
            cc1, cc2 = st.columns(2)
            if cc1.button("Confirm clear", use_container_width=True):
                vector_store.clear()
                st.session_state.confirm_clear = False
                st.rerun()
            if cc2.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()
        else:
            if st.button("Clear knowledge base", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    with st.expander("⚙ Correction settings"):
        cfg.top_k = st.slider("Chunks retrieved per attempt", 2, 10, cfg.top_k)
        cfg.min_relevant_chunks = st.slider("Verified chunks required to stop", 1, 6, cfg.min_relevant_chunks)
        cfg.max_iterations = st.slider("Max correction rounds", 0, 4, cfg.max_iterations)
        cfg.confidence_threshold = st.slider(
            "Grader confidence threshold", 0.0, 1.0, cfg.confidence_threshold, 0.05
        )
        cfg.max_context_chunks = st.slider("Max chunks sent to generator", 2, 12, cfg.max_context_chunks)
        web_ok = is_web_search_available()
        cfg.enable_web_search = st.checkbox(
            "Enable web search fallback (experimental)",
            value=cfg.enable_web_search and web_ok,
            disabled=not web_ok,
        )
        if not web_ok:
            st.caption("Install `duckduckgo-search` to enable this.")

    with st.expander("⚙ Ingestion settings"):
        cfg.chunk_size = st.slider("Chunk size (characters)", 300, 3000, cfg.chunk_size, 50)
        cfg.chunk_overlap = st.slider("Chunk overlap (characters)", 0, 500, cfg.chunk_overlap, 10)

    st.divider()
    if st.button("🗑 Reset conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# =============================================================================
# Rendering helpers for the main pane
# =============================================================================
def stamp_html(status: str) -> str:
    if status == "answered":
        return '<div class="stamp stamp-verified">✓ VERIFIED FROM SOURCE</div>'
    if status == "error":
        return '<div class="stamp stamp-flagged">⚠ CONNECTION ERROR</div>'
    return '<div class="stamp stamp-flagged">✕ UNVERIFIED — NO EVIDENCE FOUND</div>'


def grade_badge_html(grade: Grade) -> str:
    cls = "pill-verified" if grade.relevant else "pill-flagged"
    label = "VERIFIED" if grade.relevant else "REJECTED"
    return f'<span class="pill {cls}">{label} · {grade.confidence:.2f}</span>'


def render_source_card(i: int, chunk: Dict) -> None:
    meta = chunk.get("metadata", {})
    source = html_lib.escape(str(meta.get("source", "unknown")))
    page = meta.get("page")
    loc = f" · p.{page}" if page else ""
    conf = chunk.get("confidence")
    conf_txt = f"{conf:.2f}" if conf is not None else "n/a"
    preview = chunk["text"][:500]
    preview += "…" if len(chunk["text"]) > 500 else ""
    st.markdown(
        f"""<div class="evidence-card">
  <div class="evidence-head">
    <span class="evidence-tag">[{i}]</span>
    <span class="evidence-source">{source}{loc}</span>
    <span class="evidence-confidence">confidence {conf_txt}</span>
  </div>
  <div class="evidence-body">{html_lib.escape(preview)}</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_trace(events: List[Dict]) -> None:
    if not events:
        st.caption("No trace recorded.")
        return
    for event in events:
        t = event["type"]
        if t == "retrieval":
            q = html_lib.escape(event["query"])
            st.markdown(f"**Attempt {event['iteration'] + 1} — retrieval** for query: *{q}*")
            if not event["results"]:
                st.caption("No chunks returned from the vector store.")
        elif t == "grading":
            for chunk, grade in event["graded"]:
                meta = chunk.get("metadata", {})
                src = html_lib.escape(str(meta.get("source", "unknown")))
                preview = html_lib.escape(chunk["text"][:140])
                reason = html_lib.escape(grade.reason or "")
                st.markdown(
                    f'{grade_badge_html(grade)} `{src}` — {preview}…<br>'
                    f'<span class="mono-note">reason: {reason}</span>',
                    unsafe_allow_html=True,
                )
        elif t == "rewrite":
            reason = html_lib.escape(event["reason"])
            new_q = html_lib.escape(event["new_query"])
            st.markdown(f"↻ **Rewriting query** ({reason}) → *{new_q}*")
        elif t == "web_search":
            used = "used" if event["used"] else "found nothing usable"
            st.markdown(f"🌐 Web search fallback: {used} ({len(event['results'])} raw result(s))")
        st.markdown("---")


def render_assistant_turn(turn: Dict) -> None:
    st.markdown(stamp_html(turn.get("status", "answered")), unsafe_allow_html=True)
    st.markdown(turn["content"])

    if turn.get("status") == "answered":
        c1, c2, c3 = st.columns(3)
        c1.metric("Correction rounds", turn.get("iterations_used", 1))
        c2.metric("Verified sources", len(turn.get("sources", [])))
        c3.metric("Web fallback", "Used" if turn.get("web_used") else "Not used")

        sources = turn.get("sources", [])
        with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
            for i, s in enumerate(sources, start=1):
                render_source_card(i, s)

    with st.expander("🧪 Pipeline trace", expanded=False):
        render_trace(turn.get("trace", []))


def run_pipeline_with_live_status(question: str) -> Dict:
    pipeline = CorrectiveRAGPipeline(vector_store, client, cfg)
    trace_events: List[Dict] = []
    final: Optional[Dict] = None

    with st.status("Working through the corrective pipeline…", expanded=True) as box:
        try:
            for event in pipeline.run_stream(question):
                trace_events.append(event)
                etype = event["type"]
                if etype == "status":
                    box.write(event["message"])
                elif etype == "retrieval":
                    box.write(
                        f"Retrieved {len(event['results'])} candidate chunk(s) for: "
                        f"\u201c{event['query']}\u201d"
                    )
                elif etype == "grading":
                    n_rel = sum(1 for _, g in event["graded"] if g.relevant)
                    box.write(f"Grading complete — {n_rel}/{len(event['graded'])} chunk(s) marked relevant.")
                elif etype == "rewrite":
                    box.write(f"Query rewritten \u2192 \u201c{event['new_query']}\u201d")
                elif etype == "web_search":
                    box.write(
                        "Web search fallback "
                        + ("found usable results." if event["used"] else "found nothing usable.")
                    )
                elif etype == "final":
                    final = event

            if final and final["status"] == "answered":
                box.update(label="Answer generated from verified context.", state="complete")
            else:
                box.update(label="No verified evidence found.", state="error")
        except OllamaError as e:
            box.update(label="Pipeline stopped — connection error.", state="error")
            final = {
                "status": "error",
                "answer": f"⚠️ Lost the connection to Ollama mid-run: {e}",
                "sources": [],
                "web_used": False,
                "iterations_used": 0,
            }

    assert final is not None
    return {
        "role": "assistant",
        "content": final["answer"],
        "status": final["status"],
        "sources": final.get("sources", []),
        "web_used": final.get("web_used", False),
        "iterations_used": final.get("iterations_used", 0),
        "trace": trace_events,
    }


# =============================================================================
# Main pane
# =============================================================================
st.markdown('<div class="app-title">Evidence Desk</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Corrective RAG System · answers verified before they\'re shown</div>', unsafe_allow_html=True)

tab_ask, tab_kb, tab_how = st.tabs(["💬  Ask", "📚  Knowledge Base", "🔎  How It Works"])

# ----------------------------------------------------------------- Ask tab
with tab_ask:
    kb_empty = vector_store.count() == 0
    models_missing = not cfg.llm_model or not cfg.embed_model

    if kb_empty:
        st.info("Your knowledge base is empty. Upload documents from the sidebar to get started.")
    elif models_missing:
        st.warning("Select a language model and an embedding model in the sidebar before asking questions.")

    for turn in st.session_state.chat_history:
        avatar = "🧑" if turn["role"] == "user" else "🗂️"
        with st.chat_message(turn["role"], avatar=avatar):
            if turn["role"] == "user":
                st.markdown(turn["content"])
            else:
                render_assistant_turn(turn)

    ready = available and not kb_empty and not models_missing
    question = st.chat_input(
        "Ask a question about your documents…",
        disabled=not ready,
    )

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🗂️"):
            result = run_pipeline_with_live_status(question)
            render_assistant_turn(result)
        st.session_state.chat_history.append(result)

# ----------------------------------------------------------- Knowledge Base tab
with tab_kb:
    st.subheader("Ingested documents")
    kb_sources = vector_store.list_sources()
    if not kb_sources:
        st.info("No documents ingested yet. Use the sidebar to upload PDF, DOCX, TXT, or MD files.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Documents", len(kb_sources))
        c2.metric("Total chunks", sum(kb_sources.values()))
        st.divider()
        for src, count in sorted(kb_sources.items()):
            cols = st.columns([5, 1, 1])
            cols[0].markdown(f"**{src}**")
            cols[1].markdown(f"`{count} chunks`")
            if cols[2].button("Remove", key=f"kb_tab_remove_{src}"):
                vector_store.delete_source(src)
                st.rerun()

# ----------------------------------------------------------- How It Works tab
with tab_how:
    st.subheader("The corrective loop")
    st.caption(
        "Retrieval isn't trusted by default — every chunk is graded before it's "
        "allowed anywhere near the final answer."
    )

    steps = [
        ("01", "Ingest", "Documents are uploaded, text is extracted (PDF / DOCX / TXT / MD), cleaned of "
                          "artifacts like hyphenation breaks and stray page numbers, then split into "
                          "overlapping chunks with source and page metadata attached."),
        ("02", "Embed & store", "Each chunk is embedded by a local Ollama embedding model and stored in a "
                                 "persistent Chroma vector database — no cloud, no API key."),
        ("03", "Retrieve", "The user's question is embedded and the top-k most similar chunks are pulled "
                            "from the vector store."),
        ("04", "Grade", "An LLM grades every retrieved chunk against the original question: relevant or "
                         "not, with a confidence score and a short reason. Nothing is trusted just because "
                         "it was retrieved."),
        ("05", "Correct", "If too few chunks are verified, the question is rewritten to be more specific "
                           "or better suited to vector search, and retrieval runs again — up to a "
                           "configurable number of rounds."),
        ("06", "Fall back (optional)", "If the document knowledge base still comes up empty, an optional "
                                        "web search can be tried as a last resort — its results are graded "
                                        "exactly like document chunks before being trusted."),
        ("07", "Generate", "The answer is written strictly from verified chunks, with inline [n] citations. "
                            "The model is instructed never to fall back on outside knowledge."),
        ("08", "Verify or decline", "If nothing ever clears the relevance bar, the system says so plainly "
                                     "instead of guessing — an honest 'no evidence found' beats a fluent "
                                     "hallucination."),
    ]
    for num, title, body in steps:
        st.markdown(
            f"""<div class="process-row">
  <div class="process-num">{num}</div>
  <div>
    <div class="process-title">{title}</div>
    <div class="process-body">{body}</div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )
