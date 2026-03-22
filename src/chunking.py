from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List, Sequence, Tuple


@dataclass
class TextChunk:
    """Container for one chunk of source text.

    Example:
        c = TextChunk("doc_chunk_0", "docs/a.txt", "RAG combines retrieval and generation.")
    """

    chunk_id: str
    source: str
    text: str


def read_text_files(
    docs_dir: Path, extensions: Sequence[str] = (".txt", ".md")
) -> List[Tuple[str, str]]:
    """Read supported text files and return `(source_path, content)` tuples.

    Example:
        docs = read_text_files(Path("sample_docs"))
        # [("sample_docs/a.txt", "some content"), ...]
    """
    if not docs_dir.exists() or not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    documents: List[Tuple[str, str]] = []
    for file_path in sorted(docs_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in extensions:
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            documents.append((str(file_path), text))
    return documents


def chunk_document(
    text: str,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[TextChunk]:
    """Split a single document into sentence-aware overlapping chunks.

    Example:
        chunks = chunk_document(
            text="Sentence one. Sentence two. Sentence three.",
            source="docs/rag.txt",
            chunk_size=25,
            chunk_overlap=8,
        )
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[TextChunk] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > chunk_size:
            if current:
                chunk_text = " ".join(current).strip()
                chunks.append(
                    TextChunk(
                        chunk_id=f"{Path(source).stem}_chunk_{len(chunks)}",
                        source=source,
                        text=chunk_text,
                    )
                )
                current, current_len = _overlap_tail(current, chunk_overlap)

            for piece in _hard_split(sentence, chunk_size):
                chunks.append(
                    TextChunk(
                        chunk_id=f"{Path(source).stem}_chunk_{len(chunks)}",
                        source=source,
                        text=piece,
                    )
                )
            continue

        projected = current_len + len(sentence) + (1 if current else 0)
        if projected <= chunk_size:
            current.append(sentence)
            current_len = projected
            continue

        chunk_text = " ".join(current).strip()
        if chunk_text:
            chunks.append(
                TextChunk(
                    chunk_id=f"{Path(source).stem}_chunk_{len(chunks)}",
                    source=source,
                    text=chunk_text,
                )
            )
        current, current_len = _overlap_tail(current, chunk_overlap)
        current.append(sentence)
        current_len = len(" ".join(current))

    if current:
        chunk_text = " ".join(current).strip()
        if chunk_text:
            chunks.append(
                TextChunk(
                    chunk_id=f"{Path(source).stem}_chunk_{len(chunks)}",
                    source=source,
                    text=chunk_text,
                )
            )
    return chunks


def chunk_documents(
    documents: Iterable[Tuple[str, str]],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[TextChunk]:
    """Chunk multiple documents by calling `chunk_document` for each item.

    Example:
        docs = [("docs/a.txt", "Hello world. More text.")]
        chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=10)
    """
    all_chunks: List[TextChunk] = []
    for source, text in documents:
        all_chunks.extend(
            chunk_document(
                text=text,
                source=source,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return all_chunks


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation boundaries.

    Example:
        _split_into_sentences("A. B? C!")
        # ["A.", "B?", "C!"]
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return re.split(r"(?<=[.!?])\s+", normalized)


def _hard_split(text: str, chunk_size: int) -> List[str]:
    """Split long text into fixed-size pieces by character length.

    Example:
        _hard_split("abcdefgh", 3)
        # ["abc", "def", "gh"]
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _overlap_tail(sentences: List[str], chunk_overlap: int) -> Tuple[List[str], int]:
    """Keep a tail of sentences whose total length fits `chunk_overlap`.

    Returns:
        `(kept_sentences, total_chars)`

    Example:
        _overlap_tail(["One.", "Two.", "Three."], chunk_overlap=10)
    """
    if chunk_overlap == 0 or not sentences:
        return [], 0

    kept: List[str] = []
    total = 0
    for sentence in reversed(sentences):
        next_total = total + len(sentence) + (1 if kept else 0)
        if next_total > chunk_overlap:
            break
        kept.insert(0, sentence)
        total = next_total
    return kept, len(" ".join(kept))
