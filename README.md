# Mr. HelpMate AI — RAG with ChromaDB, and Gemini

This repository implements Mr. HelpMate AI — a PDF-focused Retrieval-Augmented Generation (RAG) system that:
- Ingests PDFs, chunks text with overlap, generates Google Gemini embeddings, and stores them in ChromaDB (Cloud).
- Serves a RAG chatbot workflow that moderates inputs, validates intent, retrieves with a cache-first strategy, re-ranks using a cross-encoder, and generates grounded answers with Gemini, while maintaining chat history.

Repository: [aydiegithub/mr-helpmate-ai](https://github.com/aydiegithub/mr-helpmate-ai)


## Table of Contents
- Features
- Workflows
- Project Structure
- Setup
- Environment Variables
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
- Logging hooks across components.



## Workflows

1) Document Ingestion and Embedding
- Extract full text from PDF -> str (no list) using pdfplumber.
- Chunk text with overlap using LangChain’s RecursiveCharacterTextSplitter.
- Embed each chunk via Google Gemini embeddings.
- Upsert documents, embeddings, and deterministic IDs into a ChromaDB Cloud collection.

2) RAG Chatbot
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
  uv add -r requirements.txt
  uv run app.py
  ```

- With pip:
  ```bash
  pip install -r requirements.txt
  python app.py
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



### 2) Run the RAG Chatbot (interactive loop)

This mirrors the basic loop you shared:

```python
from src.backend.generation_layer import ChatCompletion
from src.artifacts import SystemInstruction
from src.backend.intent_moderation_check import (IntentCheck, ModerationCheck)
from src.database.chromadb_vectorstore import VectorStore
from src.backend.reranker import Reranker

mod_chk = ModerationCheck()
int_cnf = IntentCheck()
chat_completion = ChatCompletion()
vector_store = VectorStore()
reranker = Reranker()

messages = [
    {"role": "system", "content": SystemInstruction.prompt}
]

while True:
    input_message = input("User: ")
    if input_message in ["exit", "bye", "end"]:
        print("Thank you for your time, hope I helped. Bye!")
        print("Chat Terminated....")
        break

    if mod_chk.check_moderation(input_message=input_message):
        print("Your Conversation has been flagged!, restarting the conversation.")
        continue

    if int_cnf.check_intent(input_message=input_message):
        top_10_documents = vector_store.query_from_db(query=input_message, top_k=10)

        reranked_top_3 = reranker.rerank_documents(
            documents=top_10_documents['documents'],
            query=input_message,
            top_k=3
        )

        reranked_context = ""
        for score, doc in reranked_top_3:
            reranked_context += doc + "\n\n\n"

        messages.append({
            "role": "user",
            "content": input_message + "###context:\n" + reranked_context
        })

        ai_response = chat_completion.chat_completion(messages=messages)

        if mod_chk.check_moderation(input_message=ai_response):
            print("Your Conversation has been flagged!, restarting the conversation.")
            continue

        print("Assistant:", ai_response, end="\n\n")
    else:
        print("Your question is out of scope or context, please ask the right question related to the domain.")
        continue
```

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



## Notes and Best Practices
- Ensure `GOOGLE_API_KEY` and Chroma Cloud credentials are set before running ingestion or chat.
- Tune chunk size/overlap for your corpus; 1200/250 works well for dense PDFs.
- Cache distance threshold controls “similar enough” semantics for prior queries; start around 0.2 and adjust after observing hits.
- Consider persisting rerank outputs for popular queries if latency matters.
- Log at info/warn levels to diagnose retrieval vs. cache vs. re-rank timing bottlenecks.



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



## License
See [LICENSE](https://github.com/aydiegithub/mr-helpmate-ai/blob/main/LICENSE).



## Project Info
- Author: Aditya (Aydie) Dinesh K
- Email: developer@aydie.in
- Website: https://www.aydie.in/
- Projects: https://projects.aydie.in/
- Completed On: Oct, 8th 2025
- Repository: [aydiegithub/mr-helpmate-ai](https://github.com/aydiegithub/mr-helpmate-ai)