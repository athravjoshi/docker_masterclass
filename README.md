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
python -m pip install -r requirement.txt
```

4. Add your Gemini key in `.env` (inside `docker_masterclass/.env`):

```env
GEMINI_API_KEY=your_api_key_here
```

Optional model settings:

```env
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_GENERATION_MODEL=gemini-2.5-flash
```

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
