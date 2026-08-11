"""
Thin wrapper around the local Ollama REST API.

This is the *only* place in the project that talks to a model. Deliberately
plain `requests` calls against http://localhost:11434 (or wherever the user
points it) instead of a cloud SDK, because the whole point of this project
is: no API key, no cloud dependency, everything runs on the user's machine.

Endpoints used:
  GET  /api/tags        -> list locally available models
  POST /api/chat         -> chat completion (used for grading, rewriting,
                             and answer generation)
  POST /api/embeddings   -> single-text embedding vector
"""

from __future__ import annotations

from typing import Dict, List

import requests


class OllamaError(RuntimeError):
    """Raised for any failure talking to the Ollama server."""


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 180):
        self.host = host.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> List[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            data = r.json()
            return sorted(m["name"] for m in data.get("models", []))
        except requests.exceptions.RequestException:
            return []

    # ------------------------------------------------------------------ #
    @staticmethod
    def _server_error_detail(response: requests.Response) -> str:
        """Ollama puts the real reason in the JSON body's "error" field;
        requests' default HTTPError message discards it. Surface it."""
        try:
            body = response.json()
            if isinstance(body, dict) and "error" in body:
                return str(body["error"])
        except ValueError:
            pass
        return (response.text or "").strip()[:500]

    def embed(self, text: str, model: str) -> List[float]:
        if not model:
            raise OllamaError("No embedding model selected.")
        try:
            r = requests.post(
                f"{self.host}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as e:
            raise OllamaError(
                f"Could not reach Ollama at {self.host} for embeddings ({e})."
            ) from e
        if not r.ok:
            detail = self._server_error_detail(r)
            raise OllamaError(
                f"Ollama rejected the embedding request ({r.status_code}): {detail}"
            )
        data = r.json()
        if "embedding" not in data:
            raise OllamaError(f"Ollama returned no embedding: {data}")
        return data["embedding"]

    def embed_batch(self, texts: List[str], model: str) -> List[List[float]]:
        # Ollama's /api/embeddings endpoint is single-text; loop explicitly
        # so callers can show per-item progress.
        return [self.embed(t, model) for t in texts]

    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        if not model:
            raise OllamaError("No language model selected.")
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            r = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
        except requests.exceptions.RequestException as e:
            raise OllamaError(
                f"Could not reach Ollama at {self.host} for chat ({e})."
            ) from e
        if not r.ok:
            detail = self._server_error_detail(r)
            raise OllamaError(
                f"Ollama rejected the chat request ({r.status_code}): {detail}"
            )
        data = r.json()
        message = data.get("message", {})
        content = message.get("content")
        if content is None:
            raise OllamaError(f"Ollama returned no content: {data}")
        return content
