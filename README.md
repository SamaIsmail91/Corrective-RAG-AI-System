# Evidence Desk — a Corrective RAG System

A Retrieval-Augmented Generation system that doesn't trust its own retrieval.
Every chunk pulled from your documents is **graded for relevance by an LLM**
before it's allowed near an answer. If the retrieval was weak, the system
**rewrites the question and tries again** (up to a configurable number of
rounds) instead of quietly answering from noise. If nothing ever checks out,
it **says so plainly** instead of guessing.

Everything runs against a local [Ollama](https://ollama.com) server — there
is no OpenAI/Anthropic/cloud API key anywhere in this project.

```
Retrieve → Grade → weak? → Rewrite → Retrieve again → Grade
                                                          │
                              enough verified evidence? ──┤
                                                          │
                          yes → Generate (cited, from verified context only)
                          no  → say "no evidence found" — never hallucinate
```

## Features

- **Ingestion pipeline** — upload PDF / DOCX / TXT / MD, extract text,
  clean common artifacts (hyphenation breaks, stray page numbers), and split
  into overlapping chunks with source + page metadata.
- **Local embeddings & vector store** — embeddings come from an Ollama
  embedding model (e.g. `nomic-embed-text`) and are stored in a persistent
  [Chroma](https://www.trychroma.com/) database on disk.
- **Relevance grading** — every retrieved chunk is graded `relevant` /
  `irrelevant` with a confidence score and a reason, by the LLM.
- **Query rewriting & re-retrieval** — when grading comes back weak, the
  question is rewritten (told *why* the previous attempt failed) and
  retrieval runs again.
- **Optional web-search fallback** — if the document knowledge base still
  has nothing after correction, an opt-in DuckDuckGo search can be tried as
  a last resort; results go through the same grading step before they can
  be used.
- **Grounded generation only** — the answer model is instructed to use only
  verified context, cite it inline as `[1]`, `[2]`, and explicitly say what's
  missing rather than fill gaps from general knowledge.
- **Full transparency** — every run shows a live, step-by-step trace
  (retrieval attempts, per-chunk grades and reasons, rewrites, generation)
  plus a sources panel with confidence scores.
- **Handles the "I don't know" case** — an honest "no evidence found"
  response instead of a hallucinated one.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- At least one Ollama chat model and one Ollama embedding model pulled

## Setup

**1. Install and start Ollama**, then pull a chat model and an embedding
model:

```bash
ollama pull llama3.1            # or llama3.2 / qwen2.5 / mistral / phi3
ollama pull nomic-embed-text    # or mxbai-embed-large / all-minilm
ollama serve                    # if it isn't already running as a service
```

**2. Install Python dependencies:**

```bash
pip install -r requirements.txt
```

> The `duckduckgo-search` package (used only for the optional web-search
> fallback) is listed but not required — the app runs fine without it, the
> web-search toggle just stays disabled.

**3. Run the app:**

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Using it

1. In the sidebar, confirm Ollama shows **Connected** and pick your language
   model and embedding model from the dropdowns.
2. Upload one or more documents and click **Process documents**.
3. Ask a question in the **Ask** tab. Watch the live pipeline trace as it
   retrieves, grades, and (if needed) corrects itself.
4. Expand **Sources** to see exactly which chunks the answer is grounded in,
   with per-chunk confidence scores. Expand **Pipeline trace** to see the
   full retrieve/grade/rewrite history for that question.
5. Tune correction behavior (top-k, how many verified chunks are required,
   max correction rounds, grader confidence threshold) under **⚙ Correction
   settings** in the sidebar.

## Project layout

```
app.py                     Streamlit UI
config.py                  Defaults + the mutable RuntimeConfig used everywhere
src/
  text_splitter.py         Dependency-free recursive character splitter
  document_loader.py       Extraction (PDF/DOCX/TXT/MD) + cleaning + chunking
  ollama_client.py         Thin REST wrapper around the local Ollama server
  vector_store.py          Chroma-backed persistent vector store
  ingestion_service.py     Ties loading + embedding + storage together
  grader.py                LLM-as-judge relevance grading
  query_rewriter.py        Corrective query rewriting
  generator.py             Grounded, cited answer generation
  web_search.py            Optional, best-effort web-search fallback
  pipeline.py              The Corrective RAG orchestrator (streams a trace)
data/uploads/               Temp storage for uploaded files during ingestion
vectordb/                   Persistent Chroma database (created on first run)
```

## Notes on the corrective loop

Grading always evaluates chunks against the **original** question — only the
*retrieval* query changes between rounds. This keeps "relevant" meaning the
same thing throughout a run, even as the search query itself evolves. The
loop stops as soon as enough verified chunks accumulate, or after the
configured number of correction rounds, whichever comes first — so a bad
first retrieval never silently caps the number of chances the system gets to
find real evidence, but a good first retrieval also doesn't get delayed by
unnecessary extra rounds.

## Troubleshooting

- **"Not reachable" in the sidebar** — make sure `ollama serve` is running
  and the Server URL matches (default `http://localhost:11434`).
- **No models in the dropdowns** — run `ollama pull llama3.1` and
  `ollama pull nomic-embed-text` (or your preferred models), then hit the
  refresh (↻) button next to the connection status.
- **A PDF ingests with 0 chunks** — it's likely a scanned/image-only PDF
  with no extractable text layer; OCR isn't included in this project.
