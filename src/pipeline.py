"""
The Corrective RAG orchestrator.

`run_stream()` is a generator so the Streamlit UI can render each step
(retrieve -> grade -> [rewrite -> retrieve -> grade]* -> [web search] ->
generate) live, instead of the user staring at a blank spinner. Every
yielded event is a plain dict with a "type" key; app.py switches on it.

Loop shape, per the Corrective-RAG idea:
  1. Retrieve top-k chunks for the current query.
  2. Grade every chunk against the ORIGINAL question (grading always uses
     the original question — only retrieval uses the rewritten one — so
     "relevant" keeps a stable meaning across iterations).
  3. Keep chunks graded relevant and above the confidence threshold.
  4. Stop correcting once enough verified chunks have accumulated, or once
     max_iterations is reached.
  5. Otherwise, rewrite the query with the reason the last attempt was weak,
     and go again.
  6. If nothing verified survives and web search is enabled, try that as a
     last resort, grading those results too.
  7. Generate the answer strictly from verified context, or return the
     "unanswered" message if there's still nothing to work with.
"""

from __future__ import annotations

from typing import Dict, Iterator, List

from config import RuntimeConfig, UNANSWERED_MESSAGE
from src.generator import AnswerGenerator
from src.grader import RelevanceGrader
from src.ollama_client import OllamaClient
from src.query_rewriter import QueryRewriter
from src.vector_store import VectorStore
from src.web_search import web_search as run_web_search


class CorrectiveRAGPipeline:
    def __init__(self, vector_store: VectorStore, client: OllamaClient, cfg: RuntimeConfig):
        self.vs = vector_store
        self.client = client
        self.cfg = cfg
        self.grader = RelevanceGrader(client, cfg.llm_model)
        self.rewriter = QueryRewriter(client, cfg.llm_model)
        self.generator = AnswerGenerator(client, cfg.llm_model)

    # ------------------------------------------------------------------ #
    def run_stream(self, question: str) -> Iterator[Dict]:
        verified: List[Dict] = []
        seen_ids = set()
        current_query = question
        reason = ""

        yield {"type": "start", "question": question}

        for iteration in range(self.cfg.max_iterations + 1):
            yield {
                "type": "status",
                "message": f"Retrieving candidates (attempt {iteration + 1})…",
            }
            q_embedding = self.client.embed(current_query, self.cfg.embed_model)
            retrieved = self.vs.similarity_search(q_embedding, k=self.cfg.top_k)
            yield {
                "type": "retrieval",
                "iteration": iteration,
                "query": current_query,
                "results": retrieved,
            }

            if not retrieved:
                grades = []
            else:
                yield {
                    "type": "status",
                    "message": f"Grading {len(retrieved)} retrieved chunk(s) for relevance…",
                }
                grades = self.grader.grade_batch(question, retrieved)

            graded_pairs = list(zip(retrieved, grades))
            yield {"type": "grading", "iteration": iteration, "graded": graded_pairs}

            new_count = 0
            for chunk, grade in graded_pairs:
                if (
                    grade.relevant
                    and grade.confidence >= self.cfg.confidence_threshold
                    and chunk["id"] not in seen_ids
                ):
                    seen_ids.add(chunk["id"])
                    verified.append(
                        {
                            **chunk,
                            "confidence": grade.confidence,
                            "grade_reason": grade.reason,
                            "found_in_iteration": iteration,
                        }
                    )
                    new_count += 1

            if len(verified) >= self.cfg.min_relevant_chunks:
                break
            if iteration == self.cfg.max_iterations:
                reason = "reached the maximum number of correction attempts"
                break

            reason = (
                "no relevant passages were found"
                if new_count == 0
                else "only partial relevance was found"
            )
            yield {
                "type": "status",
                "message": f"Retrieval was weak ({reason}) — rewriting the query…",
            }
            current_query = self.rewriter.rewrite(question, current_query, reason)
            yield {"type": "rewrite", "iteration": iteration, "new_query": current_query, "reason": reason}

        web_used = False
        web_raw: List[Dict] = []
        if len(verified) < self.cfg.min_relevant_chunks and self.cfg.enable_web_search:
            yield {
                "type": "status",
                "message": "Knowledge base still insufficient — trying a web search fallback…",
            }
            web_raw = run_web_search(question)
            if web_raw:
                web_grades = self.grader.grade_batch(question, web_raw)
                for chunk, grade in zip(web_raw, web_grades):
                    if grade.relevant and grade.confidence >= self.cfg.confidence_threshold:
                        verified.append(
                            {
                                **chunk,
                                "confidence": grade.confidence,
                                "grade_reason": grade.reason,
                                "found_in_iteration": "web",
                            }
                        )
                        web_used = True
            yield {"type": "web_search", "results": web_raw, "used": web_used}

        if not verified:
            yield {
                "type": "final",
                "status": "unanswered",
                "answer": UNANSWERED_MESSAGE,
                "sources": [],
                "web_used": web_used,
                "iterations_used": iteration + 1,
            }
            return

        yield {"type": "status", "message": "Generating an answer from verified context…"}
        verified.sort(key=lambda c: c["confidence"], reverse=True)
        top_context = verified[: self.cfg.max_context_chunks]
        answer = self.generator.generate(question, top_context)

        yield {
            "type": "final",
            "status": "answered",
            "answer": answer,
            "sources": top_context,
            "web_used": web_used,
            "iterations_used": iteration + 1,
        }
