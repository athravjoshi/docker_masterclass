from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Protocol, Sequence

import requests


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class GeminiAPIError(RuntimeError):
    """Base error for Gemini API failures."""


class GeminiQuotaExceededError(GeminiAPIError):
    """Raised when account quota is exhausted."""


class GeminiRateLimitError(GeminiAPIError):
    """Raised when transient API throttling occurs."""


class Embedder(Protocol):
    """Minimal embedding interface used by the RAG app."""

    def embed_text(self, text: str) -> List[float]:
        """Embed one chunk/document."""

    def embed_query(self, text: str) -> List[float]:
        """Embed one query."""

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        """Batch embed many chunks/documents."""


@dataclass
class SearchResult:
    """Represents one retrieved chunk and its similarity score."""

    chunk_id: str
    source: str
    text: str
    score: float


class GeminiClient:
    """Small Gemini REST client for answer generation only."""

    def __init__(
        self,
        api_key: str,
        generation_model: str = "gemini-2.5-flash",
        timeout_seconds: int = 45,
        max_retries: int = 2,
        backoff_seconds: float = 1.5,
    ) -> None:
        if not api_key:
            raise ValueError("Gemini API key is required.")

        self.api_key = api_key
        self.generation_model = _normalize_model_name(generation_model)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.backoff_seconds = max(0.1, backoff_seconds)

    def generate_answer(self, prompt: str) -> str:
        """Generate a final natural language answer from a prepared prompt."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        return self._generate_content(prompt)

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a POST request and return parsed JSON or raise a clear error."""
        last_error: GeminiAPIError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url, json=payload, timeout=self.timeout_seconds
                )
            except requests.RequestException as error:
                if attempt >= self.max_retries:
                    raise GeminiAPIError(
                        f"Gemini API request failed: {error}"
                    ) from error
                self._sleep_before_retry(attempt, retry_after_seconds=None)
                continue

            if response.ok:
                return response.json()

            status = response.status_code
            details = _extract_error_details(response)
            message = f"Gemini API error ({status}): {details}"

            if status == 404 and "generateContent" in url:
                message += (
                    " | Hint: check GEMINI_GENERATION_MODEL. "
                    "Recommended values: gemini-2.5-flash or gemini-2.0-flash"
                )

            if status == 429 and _looks_like_quota_exhausted(details):
                raise GeminiQuotaExceededError(
                    "Gemini API quota exhausted (HTTP 429). "
                    "Enable billing or wait for quota reset, then retry."
                )

            if status in RETRIABLE_STATUS_CODES and attempt < self.max_retries:
                last_error = (
                    GeminiRateLimitError(message)
                    if status == 429
                    else GeminiAPIError(message)
                )
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                self._sleep_before_retry(attempt, retry_after_seconds=retry_after)
                continue

            if status == 429:
                raise GeminiRateLimitError(
                    "Gemini API rate limit hit (HTTP 429). Retry after a short wait."
                )

            raise GeminiAPIError(message)

        raise GeminiAPIError(f"Gemini API request failed after retries: {last_error}")

    def _sleep_before_retry(
        self, attempt: int, retry_after_seconds: float | None
    ) -> None:
        if retry_after_seconds is not None and retry_after_seconds >= 0:
            wait_seconds = retry_after_seconds
        else:
            wait_seconds = self.backoff_seconds * (2**attempt)
        time.sleep(wait_seconds)

    def _generate_content(self, prompt: str) -> str:
        """Generate answer text with fallback over common generation models."""
        tried: List[str] = []
        candidates = [self.generation_model]
        for candidate in ("gemini-2.5-flash", "gemini-2.0-flash"):
            if candidate not in candidates:
                candidates.append(candidate)

        last_error: GeminiAPIError | None = None
        for model_name in candidates:
            tried.append(model_name)
            try:
                url = (
                    f"{GEMINI_API_BASE}/{model_name}:generateContent?key={self.api_key}"
                )
                payload: Dict[str, Any] = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2},
                }
                data = self._post_json(url, payload)
                candidates_data = data.get("candidates", [])
                if not candidates_data:
                    raise GeminiAPIError(
                        f"No response candidates returned for model '{model_name}'."
                    )
                parts = candidates_data[0].get("content", {}).get("parts", [])
                answer = "".join(part.get("text", "") for part in parts).strip()
                if not answer:
                    raise GeminiAPIError(
                        f"Gemini API returned an empty answer for '{model_name}'."
                    )
                if model_name != self.generation_model:
                    self.generation_model = model_name
                return answer
            except GeminiQuotaExceededError:
                raise
            except GeminiAPIError as err:
                last_error = err
                continue

        raise GeminiAPIError(
            f"Failed generation with tried models: {', '.join(tried)}. "
            f"Last error: {last_error}"
        )


class SentenceTransformerEmbedder:
    """Local embedding backend using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self._model: Any | None = None

    def embed_text(self, text: str) -> List[float]:
        return self._encode_one(text)

    def embed_query(self, text: str) -> List[float]:
        return self._encode_one(text)

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        cleaned = [self._clean_text(text) for text in texts]
        if not cleaned:
            return []
        vectors = self._get_model().encode(
            cleaned,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return [_to_float_list(vector) for vector in vectors]

    def _encode_one(self, text: str) -> List[float]:
        cleaned = self._clean_text(text)
        vector = self._get_model().encode(
            cleaned,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return _to_float_list(vector)

    def _clean_text(self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("Text to embed cannot be empty.")
        return text.strip()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Add it to requirements.txt and redeploy."
            ) from error

        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as error:  # pragma: no cover - model download/runtime errors
            raise RuntimeError(
                f"Failed to load local embedding model '{self.model_name}'. {error}"
            ) from error
        return self._model


class InMemoryVectorStore:
    """Simple RAM-based vector store for local experiments."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def add(
        self, chunk_id: str, source: str, text: str, vector: Sequence[float]
    ) -> None:
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
    """Compute cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: len(a)={len(a)} and len(b)={len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    a_norm = math.sqrt(sum(x * x for x in a))
    b_norm = math.sqrt(sum(y * y for y in b))
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return dot / (a_norm * b_norm)


def _normalize_model_name(model_name: str) -> str:
    clean = (model_name or "").strip()
    if clean.startswith("models/"):
        return clean.split("/", 1)[1]
    return clean


def _extract_error_details(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:400]

    error_obj = payload.get("error", {})
    parts: List[str] = []
    status = error_obj.get("status")
    code = error_obj.get("code")
    message = error_obj.get("message")
    if status:
        parts.append(str(status))
    if code:
        parts.append(f"code={code}")
    if message:
        parts.append(str(message))
    return " | ".join(parts)[:400] or response.text[:400]


def _looks_like_quota_exhausted(details: str) -> bool:
    normalized = (details or "").lower()
    quota_markers = (
        "exceeded your current quota",
        "quota",
        "billing",
        "resource_exhausted",
    )
    return any(marker in normalized for marker in quota_markers)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return max(0.0, seconds)


def _to_float_list(vector: Any) -> List[float]:
    if hasattr(vector, "tolist"):
        values = vector.tolist()
    else:
        values = list(vector)
    return [float(value) for value in values]
