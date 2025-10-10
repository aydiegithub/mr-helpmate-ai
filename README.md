# Mr. HelpMate AI — RAG with ChromaDB, Gemini, FastAPI, LangGraph, and Render

This repository implements Mr. HelpMate AI — a PDF-focused Retrieval-Augmented Generation (RAG) system that:
- Ingests PDFs, chunks text with overlap, generates Google Gemini embeddings, and stores them in ChromaDB (Cloud).
- Serves a RAG chatbot workflow that moderates inputs, validates intent, retrieves with a cache-first strategy, re-ranks using a cross-encoder, and generates grounded answers with Gemini, while maintaining chat history.
- Provides a web-based interface via FastAPI and WebSocket for real-time interaction.
- Uses LangGraph for orchestrating the chatbot logic in a structured graph.
- Deployed on Render for scalable hosting.

Repository: [aydiegithub/mr-helpmate-ai](https://github.com/aydiegithub/mr-helpmate-ai)


## Table of Contents
- Features
- Workflows
- Project Structure
- Setup
- Environment Variables
- Deployment on Render
- API Usage
- Usage
  - 1) Ingest PDFs to ChromaDB
  - 2) Run the RAG Chatbot (interactive loop)
- Key Modules
- Notes and Best Practices
- Troubleshooting
- License
- Project Info



## Features
- PDF text extraction (including simple table extraction) to a single consolidated string.
- Robust text chunking with overlap for better semantic coverage per chunk.
- Google Gemini embeddings for chunks and queries.
- ChromaDB Cloud vector store, with deterministic IDs for reproducibility.
- Query cache implemented as a Chroma collection: cache hit detection by embedding distance, storing top-K IDs for fast re-fetch.
- Re-ranking of retrieved candidates using a Sentence-Transformers cross-encoder for higher answer quality.
- Gemini chat completions with a strict system instruction (domain-gated, context-only).
- Safety moderation and domain intent confirmation before retrieval and generation.
- Web-based chat interface served via FastAPI and WebSocket for real-time interaction.
- Workflow orchestration using LangGraph for structured, stateful chatbot logic.
- Deployed on Render for easy and scalable access.
- Logging hooks across components.


## Workflows

1) Document Ingestion and Embedding
- Extract full text from PDF -> str (no list) using pdfplumber.
- Chunk text with overlap using LangChain’s RecursiveCharacterTextSplitter.
- Embed each chunk via Google Gemini embeddings.
- Upsert documents, embeddings, and deterministic IDs into a ChromaDB Cloud collection.

2) RAG Chatbot
- The chatbot is implemented using LangGraph, a library for building stateful workflows, ensuring robust flow control.
- Moderation check: If flagged, stop.
- Intent check: If out-of-domain, stop with a helpful notice.
- Retrieval: 
  - Cache-first: Compare query embedding to cached queries (ChromaDB collection). If similar within threshold, fetch documents by cached IDs.
  - Otherwise, query the main vector store and update the cache with returned top-K IDs.
- Re-rank: Use CrossEncoder (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2) to pick the best top-N (default 3).
- Generate: Provide reranked context + system instruction + user message to Gemini chat model. Maintain chat history.
- Optional: Re-check moderation on the generated output.


## Project Structure

The structure below reflects the repository as shown in the screenshots and the provided code:

```
.
├── .env
├── .gitignore
├── .python-version
├── index.html                 # Web chat interface
├── LICENSE
├── README.md
├── app.py
├── basic.md
├── config.py
├── main.py
├── pyproject.toml
├── requirements.txt
├── submission.ipynb
├── test.py
├── test2.py
├── testing.ipynb
├── uv.lock
├── documents/                 # Place your PDFs here for ingestion
├── logs/
└── src/
    ├── artifacts/
    │   ├── __init__.py        # SystemInstruction, IntentConfirmationPrompt, ModerationCheckPrompt
    │   └── __pycache__/
    ├── backend/
    │   ├── __init__.py
    │   ├── chunking_layer.py          # TextChunking (LangChain RecursiveCharacterTextSplitter)
    │   ├── embedding_layer.py         # VectorEmbedding (Gemini + local/persistent Chroma)
    │   ├── generation_layer.py        # ChatCompletion (Gemini chat-compatible OpenAI client)
    │   ├── intent_moderation_check.py # ModerationCheck, IntentCheck (Gemini chat)
    │   ├── pdf_retriever.py           # Extractor (pdfplumber -> str)
    │   └── reranker.py                # Reranker (Cross-Encoder + cosine backup)
    ├── constants/
    │   ├── __init__.py                # Loads env and exports constants (models, keys, names)
    │   └── __pycache__/
    ├── database/
    │   ├── __init__.py
    │   ├── chromadb_cache.py          # CacheVectorStore (Chroma Cloud collection)
    │   ├── chromadb_connection.py     # Cloud client init (api_key, tenant, database)
    │   └── chromadb_vectorstore.py    # VectorStore (upsert, query, fetch-all, delete)
    ├── frontend/                      # (Placeholder for UI integrations, e.g., Gradio)
    ├── logging/                       # (Placeholder for logging config)
    └── utils/                         # (Placeholder for helpers)
```



## Setup

1) Python
- Recommended: Python 3.10–3.12
- The repo includes `pyproject.toml` and `uv.lock` (supports [uv](https://github.com/astral-sh/uv)).

2) Install and run (choose one)

- With uv:
  ```bash
  pip install uv && uv add -r requirements.txt
  uv run main.py
  ```

- With pip:
  ```bash
  pip install -r requirements.txt
  python main.py
  ```

3) Environment
- Copy `.env` to your local environment and set the required keys below.
- Ensure `src/constants/__init__.py` loads from `.env` (e.g., via python-dotenv).



## Environment Variables

Place these in `.env` (names inferred from the code):

- `GOOGLE_API_KEY`=your_gemini_key
- `MODEL`=gemini-2.5-flash             # chat/generation model
- `EMBEDDING_MODEL`=text-embedding-004 # embedding model
- `COLLECTION_NAME`=mr_helpmate_docs   # main Chroma collection for document chunks

Chroma Cloud (required by `chromadb_connection.py`):
- `CHROMA_API_KEY`=your_chroma_cloud_api_key
- `CHROMA_TENANT`=your_tenant_id
- `CHROMA_DATABASE`=your_database_name

Cache configuration (used indirectly in code):
- `CACHE_COLLECTION_NAME`=mr_helpmate_cache
- `DISTANCE_THRESHOLD`=0.2             # example threshold for cache hit acceptance

Chunking defaults (if managed via constants; otherwise see code):
- `CHUNK_SIZE`=1200
- `CHUNK_OVERLAP`=250



## Deployment on Render

1. Sign up for a Render account at [https://render.com/](https://render.com/).
2. Create a new Web Service and connect your GitHub repository `aydiegithub/mr-helpmate-ai`.
3. Set the Build Command to: `pip install uv && uv add -r requirements.txt`
4. Set the Start Command to: `uv run main.py`
5. Configure the environment variables in the Render dashboard (copy from your `.env`).
6. Deploy the service. Once deployed, your app will be accessible at the provided Render URL, with the chat interface at the root path.

Note: Ensure `index.html` is present in the root directory for the web UI to work.



## API Usage

The application exposes a REST API and WebSocket for programmatic access or integration.

### Endpoints

- `GET /`: Serves the `index.html` file for the web-based chat interface.
- `GET /health`: Health check endpoint. Returns a JSON object with `status`, `service`, and `version`.
- `WebSocket /ws/{session_id}`: Real-time chat endpoint for interactive conversations.
  - **Send**: JSON object in the format `{"message": "your user query here"}`
  - **Receive**: JSON object `{"type": "response", "message": "AI-generated response"}` on success, or `{"type": "error", "message": "error description"}` on failure.

### Usage Example

To use the API programmatically (e.g., in a Python script or another app):

1. Choose a unique `session_id` (string) for conversation continuity.
2. Establish a WebSocket connection to `/ws/{session_id}` (use `wss://` for secure connections on Render).
3. Send user messages as JSON and process the responses.
4. For the web interface, simply visit the root URL in a browser.

Example Python code for WebSocket interaction (requires `websockets` library):

```python
import asyncio
import websockets
import json

async def chat():
    uri = "ws://localhost:8000/ws/my_session_123"  # Replace with your URL
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({"message": "Hello, how can you help with insurance?"}))
        response = await websocket.recv()
        data = json.loads(response)
        print("AI:", data["message"])

asyncio.run(chat())
```

For local development, replace the URI with `ws://localhost:8000/ws/{session_id}`. For production on Render, use the full Render URL.



### Usage

### 1) Ingest PDFs to ChromaDB

Example script (run in a REPL or a small script):

```python
from src.backend.pdf_retriever import Extractor
from src.backend.chunking_layer import TextChunking
from src.database.chromadb_vectorstore import VectorStore

# 1) Extract full text (single string)
pdf_path = "documents/your_file.pdf"
text = Extractor().content_extractor(pdf_path)

# 2) Chunk text with overlap
chunks = TextChunking().create_chunks(
    input_text=text,
    chunk_size=1200,
    overlap=250
)

# 3) Upsert chunks into Chroma (embeddings auto-generated via Gemini)
vs = VectorStore()
vs.upsert_documents(documents=chunks)

print("Ingestion complete.")
```

Notes:
- `embedding_layer.py` also provides a `VectorEmbedding` class that can manage a local/persistent Chroma client; the cloud path used in `VectorStore` is recommended for production.
- Deterministic IDs are generated from content hashes (suffix `n{index}`).



### 2) Run the RAG Chatbot (interactive loop)

The chatbot can be run via the web interface (recommended) or programmatically as shown below. The logic is now orchestrated by a LangGraph for better flow control.

```python
# For programmatic use, you can invoke the graph directly (from main.py)
from main import chatbot_graph

# Example invocation
state = {
    "messages": [{"role": "system", "content": "System prompt here"}],
    "input_message": "What is insurance coverage?"
}
result = chatbot_graph.invoke(state)
print("AI Response:", result.get("ai_response"))
```

For a full interactive loop, see the WebSocket implementation in `main.py` or use the web UI.

Implementation details:
- Cache: `chromadb_cache.CacheVectorStore` stores query embeddings and metadata of top IDs (serialized under `ids_json`).
- `VectorStore.query_from_db` first checks cache for a near-duplicate query by embedding distance; on hit, fetches by IDs from the main collection.
- Re-ranking: Cross-encoder model defaults to `cross-encoder/ms-marco-MiniLM-L-6-v2`; cosine-sim rerank fallback is available.
- Generation: Uses OpenAI-compatible client pointed to Google Generative Language API `base_url` with your Gemini model.



## Key Modules

- `src/backend/pdf_retriever.py` — `Extractor.content_extractor(path) -> str`
- `src/backend/chunking_layer.py` — `TextChunking.create_chunks(text, chunk_size, overlap) -> list[str]`
- `src/database/chromadb_vectorstore.py` — `VectorStore` (upsert_documents, query_from_db, fetch_all_data, delete_documents)
- `src/database/chromadb_cache.py` — `CacheVectorStore` (check_query_in_cache, update_cache) with metadata sanitization for Chroma Cloud
- `src/database/chromadb_connection.py` — Chroma Cloud client initialization and collection management
- `src/backend/reranker.py` — Cross-encoder and cosine similarity rerankers
- `src/backend/generation_layer.py` — ChatCompletion (Gemini chat wrapper)
- `src/backend/intent_moderation_check.py` — ModerationCheck and IntentCheck (Gemini chat prompts)
- `src/artifacts/__init__.py` — SystemInstruction, IntentConfirmationPrompt, ModerationCheckPrompt
- `main.py` — FastAPI app with LangGraph chatbot integration and WebSocket endpoint



## Notes and Best Practices
- Ensure `GOOGLE_API_KEY` and Chroma Cloud credentials are set before running ingestion or chat.
- Tune chunk size/overlap for your corpus; 1200/250 works well for dense PDFs.
- Cache distance threshold controls “similar enough” semantics for prior queries; start around 0.2 and adjust after observing hits.
- Consider persisting rerank outputs for popular queries if latency matters.
- Log at info/warn levels to diagnose retrieval vs. cache vs. re-rank timing bottlenecks.
- For production, use the Render deployment for scalability and the web interface for user interaction.



## Troubleshooting
- No results from retrieval:
  - Verify the collection has been populated (run ingestion first).
  - Ensure `EMBEDDING_MODEL` matches what was used during ingestion.
- Authentication errors:
  - Double-check `.env` values, and that `src/constants` properly loads them.
- Slow responses:
  - Reduce `top_k`, or only re-rank a subset (e.g., 10 -> 3).
  - Confirm CrossEncoder model is cached locally and not repeatedly downloaded.
- Cache not hitting:
  - Lower `DISTANCE_THRESHOLD` slightly.
  - Confirm IDs are being serialized under `ids_json` and successfully parsed on read.
- Chroma Cloud errors:
  - Verify `CHROMA_TENANT` and `CHROMA_DATABASE` names and your account access.
- WebSocket connection issues:
  - Ensure the session_id is consistent for conversation continuity.
  - Check browser console for errors; ensure the WebSocket URL matches your deployment (localhost for local, Render URL for production).
- Deployment on Render fails:
  - Confirm all environment variables are set in Render.
  - Check build logs for dependency issues; ensure `uv` is compatible with Render's environment.



## License
See [LICENSE](https://github.com/aydiegithub/mr-helpmate-ai/blob/main/LICENSE).



## Project Info
- Author: Aditya (Aydie) Dinesh K
- Email: developer@aydie.in
- Website: https://www.aydie.in/
- Projects: https://projects.aydie.in/
- Completed On: Oct 10th 2025
- Repository: [aydiegithub/mr-helpmate-ai](https://github.com/aydiegithub/mr-helpmate-ai)
