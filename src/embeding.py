from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence
import warnings

import requests


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass
class SearchResult:
    """Represents one retrieved chunk and its similarity score.

    Example:
        r = SearchResult("chunk_1", "docs/a.md", "some text", 0.89)
    """

    chunk_id: str
    source: str
    text: str
    score: float


class GeminiClient:
    """Small Gemini REST client for embeddings and answer generation.

    Example:
        client = GeminiClient(api_key="your-key")
        vec = client.embed_query("What is RAG?")
    """

    def __init__(
        self,
        api_key: str,
        embedding_model: str = "gemini-embedding-001",
        generation_model: str = "gemini-pro",
        timeout_seconds: int = 45,
    ) -> None:
        """Store Gemini configuration values used by all client methods.

        Example:
            client = GeminiClient(
                api_key=os.getenv("GEMINI_API_KEY"),
                embedding_model="gemini-embedding-001",
                generation_model="gemini-2.5-flash",
            )
        """
        if not api_key:
            raise ValueError("Gemini API key is required.")

        normalized_embedding = _normalize_model_name(embedding_model)
        legacy_embedding_aliases = {
            "embedding-001",
            "text-embedding-004",
            "embedding-gecko-001",
        }
        if normalized_embedding in legacy_embedding_aliases:
            warnings.warn(
                f"Embedding model '{normalized_embedding}' may be incompatible with "
                "the current Gemini embedContent endpoint. "
                "Using 'gemini-embedding-001' instead.",
                stacklevel=2,
            )
            normalized_embedding = "gemini-embedding-001"

        normalized_generation = _normalize_model_name(generation_model)

        self.api_key = api_key
        self.embedding_model = normalized_embedding
        self.generation_model = normalized_generation
        self.timeout_seconds = timeout_seconds

    def embed_text(self, text: str) -> List[float]:
        """Embed a document/chunk for retrieval indexing.

        Uses Gemini `embedContent` with `taskType="RETRIEVAL_DOCUMENT"`.

        Example:
            vector = client.embed_text("RAG improves factuality with context.")
        """
        if not text or not text.strip():
            raise ValueError("Text to embed cannot be empty.")

        return self._embed_content(text=text, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> List[float]:
        """Embed a user query for semantic search against indexed chunks.

        Uses Gemini `embedContent` with `taskType="RETRIEVAL_QUERY"`.

        Example:
            q_vec = client.embed_query("How does Docker help deployment?")
        """
        if not text or not text.strip():
            raise ValueError("Query cannot be empty.")

        return self._embed_content(text=text, task_type="RETRIEVAL_QUERY")

    def generate_answer(self, prompt: str) -> str:
        """Generate a final natural language answer from a prepared prompt.

        Example:
            answer = client.generate_answer("Question: What is RAG? Context: ...")
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        return self._generate_content(prompt)

    def _post_json(self, url: str, payload: Dict) -> Dict:
        """Send a POST request and return parsed JSON or raise a clear error.

        Example:
            data = self._post_json("https://example.com", {"k": "v"})
        """
        response = requests.post(url, json=payload, timeout=self.timeout_seconds)
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            message = (
                f"Gemini API error ({response.status_code}): {response.text[:400]}"
            )
            if response.status_code == 404 and "embedContent" in url:
                message += (
                    " | Hint: check GEMINI_EMBEDDING_MODEL. "
                    "Recommended value: gemini-embedding-001"
                )
            if response.status_code == 404 and "generateContent" in url:
                message += (
                    " | Hint: check GEMINI_GENERATION_MODEL. "
                    "Recommended values: gemini-2.5-flash or gemini-2.0-flash"
                )
            raise RuntimeError(message) from error
        return response.json()

    def _embed_content(self, text: str, task_type: str) -> List[float]:
        """Embed text with fallback for outdated embedding model names.

        Example:
            vec = self._embed_content("hello world", "RETRIEVAL_QUERY")
        """
        tried: List[str] = []
        candidates = [self.embedding_model]
        if self.embedding_model != "gemini-embedding-001":
            candidates.append("gemini-embedding-001")

        last_error: RuntimeError | None = None
        for model_name in candidates:
            tried.append(model_name)
            try:
                url = f"{GEMINI_API_BASE}/{model_name}:embedContent?key={self.api_key}"
                payload = {
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                }
                data = self._post_json(url, payload)
                values = data.get("embedding", {}).get("values")
                if not values:
                    raise RuntimeError(
                        f"No embedding vector returned for model '{model_name}'."
                    )
                if model_name != self.embedding_model:
                    warnings.warn(
                        f"Embedding model '{self.embedding_model}' failed. "
                        f"Switched to '{model_name}'.",
                        stacklevel=2,
                    )
                    self.embedding_model = model_name
                return values
            except RuntimeError as err:
                last_error = err
                continue

        raise RuntimeError(
            f"Failed embedding with tried models: {', '.join(tried)}. "
            f"Last error: {last_error}"
        )

    def _generate_content(self, prompt: str) -> str:
        """Generate answer text with fallback over common generation models.

        Example:
            text = self._generate_content("Explain RAG in 2 lines.")
        """
        tried: List[str] = []
        candidates = [self.generation_model]
        for candidate in ("gemini-2.5-flash", "gemini-2.0-flash"):
            if candidate not in candidates:
                candidates.append(candidate)

        last_error: RuntimeError | None = None
        for model_name in candidates:
            tried.append(model_name)
            try:
                url = (
                    f"{GEMINI_API_BASE}/{model_name}:generateContent?key={self.api_key}"
                )
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2},
                }
                data = self._post_json(url, payload)
                candidates_data = data.get("candidates", [])
                if not candidates_data:
                    raise RuntimeError(
                        f"No response candidates returned for model '{model_name}'."
                    )
                parts = candidates_data[0].get("content", {}).get("parts", [])
                answer = "".join(part.get("text", "") for part in parts).strip()
                if not answer:
                    raise RuntimeError(
                        f"Gemini API returned an empty answer for '{model_name}'."
                    )
                if model_name != self.generation_model:
                    warnings.warn(
                        f"Generation model '{self.generation_model}' failed. "
                        f"Switched to '{model_name}'.",
                        stacklevel=2,
                    )
                    self.generation_model = model_name
                return answer
            except RuntimeError as err:
                last_error = err
                continue

        raise RuntimeError(
            f"Failed generation with tried models: {', '.join(tried)}. "
            f"Last error: {last_error}"
        )


class InMemoryVectorStore:
    """Simple RAM-based vector store for local experiments.

    It does not persist to disk and is rebuilt each run.

    Example:
        store = InMemoryVectorStore()
        store.add("c1", "docs/a.txt", "text", [0.1, 0.2, 0.3])
    """

    def __init__(self) -> None:
        """Initialize an empty list of vector records.

        Example:
            store = InMemoryVectorStore()
        """
        self._records: List[Dict] = []

    def add(
        self, chunk_id: str, source: str, text: str, vector: Sequence[float]
    ) -> None:
        """Insert one embedded chunk into the store.

        Example:
            store.add("chunk_0", "docs/a.txt", "some text", [0.2, 0.1])
        """
        self._records.append(
            {
                "chunk_id": chunk_id,
                "source": source,
                "text": text,
                "vector": list(vector),
            }
        )

    def search(
        self, query_vector: Sequence[float], top_k: int = 3
    ) -> List[SearchResult]:
        """Return top_k nearest chunks using cosine similarity.

        Example:
            results = store.search([0.1, 0.2, 0.3], top_k=3)
        """
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not self._records:
            return []

        scored: List[SearchResult] = []
        for record in self._records:
            score = cosine_similarity(query_vector, record["vector"])
            scored.append(
                SearchResult(
                    chunk_id=record["chunk_id"],
                    source=record["source"],
                    text=record["text"],
                    score=score,
                )
            )
        scored.sort(key=lambda row: row.score, reverse=True)
        return scored[:top_k]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Example:
        score = cosine_similarity([1, 0], [0.8, 0.2])  # ~0.97
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: len(a)={len(a)} and len(b)={len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    a_norm = math.sqrt(sum(x * x for x in a))
    b_norm = math.sqrt(sum(y * y for y in b))
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return dot / (a_norm * b_norm)


def _normalize_model_name(model_name: str) -> str:
    """Normalize model names so both `models/x` and `x` are accepted.

    Example:
        _normalize_model_name("models/gemini-embedding-001")
        # "gemini-embedding-001"
    """
    clean = (model_name or "").strip()
    if clean.startswith("models/"):
        return clean.split("/", 1)[1]
    return clean
