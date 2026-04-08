from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

try:
    from src.chunking import TextChunk, chunk_documents, read_text_files
    from src.embeding import (
        Embedder,
        GeminiClient,
        InMemoryVectorStore,
        SearchResult,
        SentenceTransformerEmbedder,
    )
except ModuleNotFoundError:
    from chunking import TextChunk, chunk_documents, read_text_files
    from embeding import (
        Embedder,
        GeminiClient,
        InMemoryVectorStore,
        SearchResult,
        SentenceTransformerEmbedder,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the RAG app.

    Example:
        python main.py --docs_dir sample_docs --query "What is RAG?"
    """
    parser = argparse.ArgumentParser(description="Simple Gemini RAG app (CLI).")
    parser.add_argument(
        "--docs_dir",
        type=Path,
        required=True,
        help="Directory with .txt or .md files to index.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Optional one-shot query. If omitted, interactive mode starts.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="How many chunks to retrieve for each query.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=800,
        help="Max characters per chunk.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=120,
        help="Characters to overlap between chunks.",
    )
    return parser.parse_args()


def load_environment() -> None:
    """Load environment variables from common .env locations.

    It tries:
    1) project_root/.env
    2) workspace_root/.env
    3) current working directory

    Example:
        load_environment()
        # Then read keys:
        # os.getenv("GEMINI_API_KEY")
    """
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    workspace_root = project_root.parent

    load_dotenv(project_root / ".env", override=True)
    load_dotenv(workspace_root / ".env", override=True)
    load_dotenv(override=True)


def build_prompt(question: str, matches: List[SearchResult]) -> str:
    """Build the final prompt using retrieved chunks as context.

    Example:
        prompt = build_prompt(
            "What is Docker?",
            [SearchResult("c1", "docs/a.md", "Docker packages apps.", 0.91)],
        )
    """
    context_parts = []
    for i, row in enumerate(matches, start=1):
        context_parts.append(
            f"[{i}] Source: {row.source}\n"
            f"Relevance: {row.score:.4f}\n"
            f"Content: {row.text}"
        )
    context = "\n\n".join(context_parts) if context_parts else "No context available."

    return (
        "You are a helpful assistant answering only from the provided context.\n"
        "If context is insufficient, say clearly what is missing.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer in a concise and clear way."
    )


def build_index(
    embedder: Embedder,
    docs_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[InMemoryVectorStore, List[TextChunk]]:
    """Create an in-memory vector index from source documents.

    Flow:
    - Read files from `docs_dir`
    - Chunk documents with overlap
    - Embed each chunk with configured embedding backend
    - Store vectors in `InMemoryVectorStore`

    Example:
        store, chunks = build_index(
            embedder=embedder,
            docs_dir=Path("sample_docs"),
            chunk_size=800,
            chunk_overlap=120,
        )
    """
    documents = read_text_files(docs_dir)
    if not documents:
        raise RuntimeError(
            f"No documents found in {docs_dir}. Add .txt or .md files and retry."
        )

    chunks = chunk_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise RuntimeError("Chunking produced no chunks.")

    store = InMemoryVectorStore()
    for idx, chunk in enumerate(chunks, start=1):
        vector = embedder.embed_text(chunk.text)
        store.add(
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            text=chunk.text,
            vector=vector,
        )
        print(f"Indexed chunk {idx}/{len(chunks)}")
    return store, chunks


def answer_question(
    generator_client: GeminiClient,
    embedder: Embedder,
    store: InMemoryVectorStore,
    question: str,
    top_k: int,
) -> tuple[str, List[SearchResult]]:
    """Answer a question using retrieval + generation.

    Steps:
    - Embed user query
    - Retrieve top_k similar chunks
    - Build prompt with retrieved context
    - Generate final answer with Gemini

    Example:
        answer, matches = answer_question(client, store, "Explain RAG", top_k=3)
    """
    query_vector = embedder.embed_query(question)
    matches = store.search(query_vector=query_vector, top_k=top_k)
    prompt = build_prompt(question, matches)
    answer = generator_client.generate_answer(prompt)
    return answer, matches


def main() -> None:
    """Run the full CLI app.

    Modes:
    - One-shot mode: if `--query` is provided
    - Interactive mode: if `--query` is omitted

    Example:
        python main.py --docs_dir sample_docs --top_k 3
    """
    load_environment()
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env."
        )

    generation_model = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")
    local_embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    client = GeminiClient(
        api_key=api_key,
        generation_model=generation_model,
    )
    embedder: Embedder = SentenceTransformerEmbedder(model_name=local_embedding_model)

    store, chunks = build_index(
        embedder=embedder,
        docs_dir=args.docs_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(
        f"\nIndex ready. Documents indexed: {len(chunks)} chunks "
        f"from folder: {args.docs_dir}\n"
    )

    if args.query.strip():
        answer, matches = answer_question(
            generator_client=client,
            embedder=embedder,
            store=store,
            question=args.query.strip(),
            top_k=args.top_k,
        )
        print("Answer:\n")
        print(answer)
        print("\nSources used:")
        for row in matches:
            print(f"- {row.source} (score={row.score:.4f})")
        return

    print("Interactive mode. Type your question, or 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        answer, matches = answer_question(
            generator_client=client,
            embedder=embedder,
            store=store,
            question=question,
            top_k=args.top_k,
        )
        print("\nAssistant:\n")
        print(answer)
        print("\nTop sources:")
        for row in matches:
            print(f"- {row.source} (score={row.score:.4f})")
        print()


if __name__ == "__main__":
    main()
