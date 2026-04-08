# docker_masterclass

This project contains a simple RAG pipeline with a Streamlit UI.

## Run the Streamlit UI (PowerShell)

1. Open PowerShell and go to the project:

```powershell
cd C:\Users\gkc\documents\docker_learn\docker_masterclass
```

2. (Recommended) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Add your Gemini key in `.env` (inside `docker_masterclass/.env`):

```env
GEMINI_API_KEY=your_api_key_here
```

Optional model settings:

```env
GEMINI_GENERATION_MODEL=gemini-2.5-flash
GEMINI_MAX_RETRIES=2
GEMINI_BACKOFF_SECONDS=1.5
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Performance note:
- First index build is slower because `sentence-transformers` downloads/loads the local model.
- Later builds in the same Streamlit process are faster because the model is cached.

5. Start Streamlit:

```powershell
python -m streamlit run app.py --server.port 8501
```

6. Open:

```text
http://localhost:8501
```

## Use the UI

1. In the sidebar, choose document source:
   - `Folder` (example: `src/sample_docs`)
   - `Upload files` (`.txt` or `.md`)
2. Click **Build / Refresh Index**.
3. Ask questions in the chat box.

## Stop the UI

Press `Ctrl + C` in the terminal running Streamlit.

## Troubleshooting 429 (Quota / Rate limit)

If you see `Gemini API quota exhausted (HTTP 429)`:

1. For Streamlit Cloud, set `GEMINI_API_KEY` in **App Settings -> Secrets**.
2. Make sure the key has available Gemini quota/billing in Google AI Studio.
3. Wait for quota reset if your current quota is exhausted.
4. Embeddings run locally with `LOCAL_EMBEDDING_MODEL`, so only answer generation uses Gemini.
5. Reduce requests by increasing chunk size or indexing fewer files.

If you see transient rate-limit 429 errors:

1. Retry after 30-60 seconds.
2. Increase retry/backoff in `.env`:

```env
GEMINI_MAX_RETRIES=4
GEMINI_BACKOFF_SECONDS=2.0
```

Note: with local embeddings, the first run may take longer because
`all-MiniLM-L6-v2` is downloaded once.
