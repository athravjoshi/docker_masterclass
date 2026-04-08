from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv


from src.chunking import TextChunk, chunk_documents, read_text_files
from src.embeding import (
    Embedder,
    GeminiClient,
    GeminiQuotaExceededError,
    GeminiRateLimitError,
    InMemoryVectorStore,
    SearchResult,
    SentenceTransformerEmbedder,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def load_environment() -> None:
    """Load env vars from common .env locations."""
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    load_dotenv(PROJECT_ROOT.parent / ".env", override=True)
    load_dotenv(override=True)


def build_prompt(question: str, matches: List[SearchResult]) -> str:
    """Build generation prompt using retrieved chunks as context."""
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


def answer_question(
    generator_client: GeminiClient,
    embedder: Embedder,
    store: InMemoryVectorStore,
    question: str,
    top_k: int,
) -> Tuple[str, List[SearchResult]]:
    """Run retrieval + generation for one question."""
    query_vector = embedder.embed_query(question)
    matches = store.search(query_vector=query_vector, top_k=top_k)
    prompt = build_prompt(question, matches)
    answer = generator_client.generate_answer(prompt)
    return answer, matches


def _read_uploaded_docs() -> List[Tuple[str, str]]:
    """Convert uploaded files into (source, text) tuples."""
    uploaded_files = st.session_state.get("uploaded_docs") or []
    documents: List[Tuple[str, str]] = []
    for file_obj in uploaded_files:
        text = file_obj.getvalue().decode("utf-8", errors="ignore").strip()
        if text:
            documents.append((file_obj.name, text))
    return documents


def _resolve_docs_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _init_state() -> None:
    if "store" not in st.session_state:
        st.session_state.store = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "client" not in st.session_state:
        st.session_state.client = None
    if "embedder" not in st.session_state:
        st.session_state.embedder = None


@st.cache_resource(show_spinner=False)
def _get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    """Cache the local embedding model across reruns for faster indexing."""
    return SentenceTransformerEmbedder(model_name=model_name)


def main() -> None:
    st.set_page_config(page_title="RAG Chat", page_icon="RAG", layout="wide")
    load_environment()
    _init_state()

    st.title("System designed by Master Athrav")
    st.write(
        "Build an index from your documents and ask questions grounded in those docs."
    )

    with st.sidebar:
        st.header("Index Settings")

        source_mode = st.radio("Document source", options=["Folder", "Upload files"])
        docs_dir_text = st.text_input("Docs folder", value="src/sample_docs")
        st.file_uploader(
            "Upload .txt/.md/.docx files",
            type=["txt", "md", "docx"],
            accept_multiple_files=True,
            key="uploaded_docs",
            disabled=source_mode != "Upload files",
        )

        chunk_size = st.slider("Chunk size", min_value=200, max_value=2000, value=800)
        max_overlap = max(0, chunk_size - 1)
        chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=min(500, max_overlap),
            value=min(120, max_overlap),
        )
        top_k = st.slider("Top K", min_value=1, max_value=10, value=3)

        build_clicked = st.button("Build / Refresh Index", use_container_width=True)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing API key. Add GEMINI_API_KEY (or GOOGLE_API_KEY) in .env.")
        st.stop()

    generation_model = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")
    local_embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
    backoff_seconds = float(os.getenv("GEMINI_BACKOFF_SECONDS", "1.5"))

    if build_clicked:
        try:
            generator_client = GeminiClient(
                api_key=api_key,
                generation_model=generation_model,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
            )
            embedder: Embedder = _get_embedder(local_embedding_model)

            if source_mode == "Folder":
                docs_path = _resolve_docs_path(docs_dir_text)
                documents = read_text_files(docs_path)
            else:
                documents = _read_uploaded_docs()

            if not documents:
                st.error("No documents found. Add .txt/.md files and try again.")
                st.stop()

            chunks: List[TextChunk] = chunk_documents(
                documents=documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if not chunks:
                st.error("Chunking produced no chunks.")
                st.stop()

            store = InMemoryVectorStore()
            progress = st.progress(0, text="Embedding chunks...")
            vectors = embedder.embed_many([chunk.text for chunk in chunks])
            if len(vectors) != len(chunks):
                raise RuntimeError(
                    "Embedding backend returned mismatched vector count."
                )
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
                store.add(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    text=chunk.text,
                    vector=vector,
                )
                progress.progress(
                    idx / len(chunks), text=f"Indexed {idx}/{len(chunks)} chunks"
                )
            progress.empty()

            st.session_state.client = generator_client
            st.session_state.embedder = embedder
            st.session_state.store = store
            st.session_state.chunks = chunks
            st.session_state.messages = []

            st.success(f"Index ready. Total chunks indexed: {len(chunks)}")
            st.caption(f"Embedding backend: local model '{local_embedding_model}'")
        except Exception as exc:
            st.error(f"Failed to build index: {exc}")
            st.stop()

    if st.session_state.store is None or st.session_state.embedder is None:
        st.info("Build the index from the sidebar, then start chatting.")
        st.stop()

    st.caption(f"Indexed chunks: {len(st.session_state.chunks)}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources used"):
                    for source in message["sources"]:
                        st.write(source)

    user_question = st.chat_input("Ask a question about your documents...")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, matches = answer_question(
                        generator_client=st.session_state.client,
                        embedder=st.session_state.embedder,
                        store=st.session_state.store,
                        question=user_question,
                        top_k=top_k,
                    )
                    st.markdown(answer)
                    source_lines = [
                        f"- {m.source} (score={m.score:.4f})" for m in matches
                    ]
                    if source_lines:
                        with st.expander("Sources used"):
                            for line in source_lines:
                                st.write(line)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": source_lines,
                        }
                    )
                except GeminiQuotaExceededError as exc:
                    st.error(f"Failed to answer question: {exc}")
                    st.info(
                        "The deployed key is out of quota. Update Streamlit secrets "
                        "with a funded Gemini key or retry after quota reset."
                    )
                except GeminiRateLimitError as exc:
                    st.error(f"Failed to answer question: {exc}")
                    st.info("Please retry in a few seconds.")
                except Exception as exc:
                    st.error(f"Failed to answer question: {exc}")


if __name__ == "__main__":
    main()
