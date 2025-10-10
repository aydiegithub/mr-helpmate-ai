# Mr. HelpMate AI — RAG with ChromaDB, LangChain, and FastAPI

This repository implements Mr. HelpMate AI — a PDF-focused Retrieval-Augmented Generation (RAG) system that:
- Ingests PDFs, chunks text with overlap, generates Google Gemini embeddings, and stores them in ChromaDB (Cloud).
- Serves a RAG chatbot workflow that moderates inputs, validates intent, retrieves with a cache-first strategy, re-ranks using a cross-encoder, and generates grounded answers with Gemini, while maintaining chat history.
- Utilizes LangChain for graph-based workflow orchestration.
- Provides a FastAPI-based REST API and WebSocket interface for real-time interactions.
- Includes a simple HTML frontend (index.html) for user interaction.
- Deployed on Render for easy access.

Repository: [aydiegithub/mr-helpmate-ai](https://github.com/aydiegithub/mr-helpmate-ai)


## Table of Contents
- Features
- Workflows
- Project Structure
- Setup
- Environment Variables
- Deployment
- Usage
  - 1) Ingest PDFs to ChromaDB
  - 2) Run the RAG Chatbot (via API/WebSocket)
  - 3) API Usage
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
- LangChain graph for structured workflow execution (moderation, intent check, retrieval, reranking, generation).
- FastAPI server with REST endpoints and WebSocket for real-time chat.
- Simple HTML frontend for interactive chat.
- Deployed on Render with automatic builds and scaling.



## Workflows

1) Document Ingestion and Embedding
- Extract full text from PDF -> str (no list) using pdfplumber.
- Chunk text with overlap using LangChain’s RecursiveCharacterTextSplitter.
- Embed each chunk via Google Gemini embeddings.
- Upsert documents, embeddings, and deterministic IDs into a ChromaDB Cloud collection.

2) RAG Chatbot (via LangChain Graph)
- Input processing: Check for greetings, moderation, and intent.
- Retrieval: 
  - Cache-first: Compare query embedding to cached queries (ChromaDB collection). If similar within threshold, fetch documents by cached IDs.
  - Otherwise, query the main vector store and update the cache with returned top-K IDs.
- Re-rank: Use CrossEncoder (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2) to pick the best top-N (default 3).
- Generate: Provide reranked context + system instruction + user message to Gemini chat model. Maintain chat history.
- Optional: Re-check moderation on the generated output.

3) FastAPI Server
- REST endpoints for health checks and serving static files (e.g., index.html).
- WebSocket endpoint for real-time chat interactions, processing messages through the LangChain graph.
- Session management for maintaining chat history per user.



## Project Structure

The structure below reflects the repository as shown in the screenshots and the provided code:

```
.
├── .env
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── index.html                 # Simple HTML frontend for chat interface
├── main.py                    # FastAPI application with LangChain graph integration
├── app.py                     # Legacy entry point (interactive loop)
├── basic.md
├── config.py
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
  pip install uv
  uv add -r requirements.txt
  uv run python main.py
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



## Deployment

The application is deployed on Render. To deploy your own instance:

1) Fork this repository to your GitHub account.
2) Sign up for a Render account at https://render.com/.
3) Create a new Web Service, linking to your forked repository.
4) Set the build command to:
   ```bash
   pip install uv && uv add -r requirements.txt
   ```
5) Set the start command to:
   ```bash
   uv run python main.py
   ```
6) Configure environment variables in Render's dashboard (same as above).
7) Deploy and access your instance via the provided URL.

Note: Ensure your ChromaDB Cloud and Gemini API keys are set securely in Render's environment variables.



## Usage

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



### 2) Run the RAG Chatbot (via API/WebSocket)

- Start the FastAPI server: `uv run python main.py` or `python main.py`.
- Access the HTML frontend at `http://localhost:8000` (or your deployed URL).
- The chat interface uses WebSocket for real-time interactions, processing through the LangChain graph.

For direct API interaction, see the API Usage section below.

The LangChain graph handles:
- Greeting detection and response.
- Input moderation.
- Intent validation.
- Document retrieval and reranking.
- Response generation with history management.



### 3) API Usage

The FastAPI app (`main.py`) provides the following endpoints:

- `GET /`: Serves the `index.html` frontend.
- `GET /health`: Health check endpoint.
  - Response: `{"status": "healthy", "service": "Mr Help Mate AI", "version": "1.0.0"}`

- `WebSocket /ws/{session_id}`: Real-time chat endpoint.
  - Connect with a unique `session_id` (e.g., user identifier).
  - Send JSON: `{"message": "Your question here"}`
  - Receive JSON: `{"type": "response", "message": "AI response"}` or `{"type": "error", "message": "Error details"}`
  - Sessions maintain chat history; greetings are handled without updating history.

Example usage with JavaScript (from index.html):

```javascript
// Establish WebSocket connection
const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);

// Send a message
ws.send(JSON.stringify({ message: "Hello, how can I get insurance quotes?" }));

// Receive response
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'response') {
        console.log('AI:', data.message);
    }
};
```

For REST API interactions, you can extend the app as needed (e.g., POST endpoints for messages), but WebSocket is recommended for chat.



## Key Modules

- `main.py` — FastAPI application with LangChain graph, WebSocket, and session management.
- `src/backend/pdf_retriever.py` — `Extractor.content_extractor(path) -> str`
- `src/backend/chunking_layer.py` — `TextChunking.create_chunks(text, chunk_size, overlap) -> list[str]`
- `src/database/chromadb_vectorstore.py` — `VectorStore` (upsert_documents, query_from_db, fetch_all_data, delete_documents)
- `src/database/chromadb_cache.py` — `CacheVectorStore` (check_query_in_cache, update_cache) with metadata sanitization for Chroma Cloud
- `src/database/chromadb_connection.py` — Chroma Cloud client initialization and collection management
- `src/backend/reranker.py` — Cross-encoder and cosine similarity rerankers
- `src/backend/generation_layer.py` — ChatCompletion (Gemini chat wrapper)
- `src/backend/intent_moderation_check.py` — ModerationCheck and IntentCheck (Gemini chat prompts)
- `src/artifacts/__init__.py` — SystemInstruction, IntentConfirmationPrompt, ModerationCheckPrompt



## Notes and Best Practices
- Ensure `GOOGLE_API_KEY` and Chroma Cloud credentials are set before running ingestion or chat.
- Tune chunk size/overlap for your corpus; 1200/250 works well for dense PDFs.
- Cache distance threshold controls “similar enough” semantics for prior queries; start around 0.2 and adjust after observing hits.
- Consider persisting rerank outputs for popular queries if latency matters.
- Log at info/warn levels to diagnose retrieval vs. cache vs. re-rank timing bottlenecks.
- For deployment, monitor Render logs for errors; ensure environment variables are secure.
- The LangChain graph provides structured, modular processing; customize nodes as needed for your domain.



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
  - Ensure the server is running and accessible (check firewall/port forwarding in deployment).
  - For local dev, use `http://localhost:8000`; for production, use the Render URL.
- LangChain graph errors:
  - Check console logs for node-specific errors (e.g., moderation or generation failures).
  - Ensure all dependencies are installed via `uv add -r requirements.txt`.



## License
See [LICENSE](https://github.com/aydiegithub/mr-helpmate-ai/blob/main/LICENSE).



## Project Info
- Author: Aditya (Aydie) Dinesh K
- Email: developer@aydie.in
- Website: https://www.aydie.in/
- Projects: https://projects.aydie.in/
- Completed On: Oct, 8th 2025
- Repository: [aydiegithub/mr-helpmate-ai](https://github.com/aydiegithub/mr-helpmate-ai)
