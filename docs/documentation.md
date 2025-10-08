# Mr. HelpMate AI - Comprehensive Documentation

## Overview

Mr. HelpMate AI is a production-ready RAG (Retrieval-Augmented Generation) system designed for insurance and banking document Q&A. The system combines PDF document ingestion, vector embeddings storage in ChromaDB Cloud, intelligent query caching, cross-encoder reranking, and Google Gemini 2.5 Flash for natural language generation. It features safety moderation, intent verification, and a sophisticated multi-stage retrieval pipeline to provide accurate, context-aware responses.

**Key Features:**
- PDF document ingestion with text and table extraction
- Text chunking with configurable overlap for semantic coherence
- Google Gemini embeddings (text-embedding-004) for semantic search
- ChromaDB Cloud for scalable vector storage
- Query caching to reduce latency and API costs
- Cross-encoder reranking for improved relevance
- Moderation and intent checks for safe, domain-specific responses
- Conversational chat interface with history management

---

## Project Structure

Complete file tree of the repository (as of current state):

```
.
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── app.py
├── config.py
├── documents/
│   ├── .DS_Store
│   └── Principal-Sample-Life-Insurance-Policy.pdf
├── main.py
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── artifacts/
│   │   └── __init__.py
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── chunking_layer.py
│   │   ├── embedding_layer.py
│   │   ├── generation_layer.py
│   │   ├── intent_moderation_check.py
│   │   ├── orchestrator.py
│   │   ├── pdf_retriever.py
│   │   └── reranker.py
│   ├── constants/
│   │   └── __init__.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── chromadb_cache.py
│   │   ├── chromadb_connection.py
│   │   └── chromadb_vectorstore.py
│   ├── frontend/
│   │   └── __init__.py
│   ├── logging/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── submission.ipynb
├── test.py
├── test2.py
├── testing.ipynb
└── uv.lock
```

**Directory Descriptions:**

- **`src/artifacts/`**: Contains prompt templates (SystemInstruction, ModerationCheckPrompt, IntentConfirmationPrompt) as dataclasses
- **`src/backend/`**: Core processing modules for chunking, embeddings, generation, moderation/intent checks, reranking, and PDF extraction
- **`src/constants/`**: Configuration constants including API keys, model names, ChromaDB settings, and thresholds
- **`src/database/`**: ChromaDB connection management, vector store operations, and query cache implementation
- **`src/logging/`**: Custom logging utility that writes timestamped logs to the `logs/` directory
- **`documents/`**: Sample PDF documents for ingestion (e.g., insurance policies)
- **`test.py`, `test2.py`**: Test scripts demonstrating ingestion, querying, reranking, and moderation/intent checks

---

## Installation

### Python Version

This project requires **Python 3.11** (as specified in `pyproject.toml`). Ensure you have Python 3.11 installed:

```bash
python --version  # Should show Python 3.11.x
```

### Setting Up a Virtual Environment

1. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   ```

2. **Activate the virtual environment:**

   - **Linux/macOS:**
     ```bash
     source .venv/bin/activate
     ```
   
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```

### Installing Dependencies

Install all required packages using either `pip` or `uv`:

**Using pip:**
```bash
pip install -r requirements.txt
```

**Using uv (recommended for faster installs):**
```bash
uv pip install -r requirements.txt
```

**Key Dependencies:**
- `google-generativeai>=0.8.0` - Google Gemini API for embeddings and chat
- `chromadb>=0.5.0` - Vector database client for ChromaDB Cloud
- `sentence-transformers>=2.2.2` - Cross-encoder models for reranking
- `pdfplumber>=0.11.0` - PDF text and table extraction
- `langchain[google-genai]>=0.3.27` - Text splitting utilities
- `langgraph>=0.2.0` - Graph-based orchestration (future use)
- `openai>=2.2.0` - OpenAI-compatible client for Gemini API
- `gradio>=4.0.0` - Web UI framework (for future interfaces)
- `pydantic>=2.0.0`, `numpy>=1.24.0`, `pandas>=2.0.0`, `tqdm>=4.66.0`, `rich>=13.0.0`, `pyyaml>=6.0.0`

---

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Google Gemini API Key
GOOGLE_API_KEY=your_google_api_key_here

# ChromaDB Cloud Configuration
CHROMA_API_KEY=your_chroma_api_key_here
CHROMA_TENANT=your_chroma_tenant
CHROMA_DATABASE=your_chroma_database
```

**Important:** Never commit the `.env` file to version control. It is listed in `.gitignore`.

### Constants (src/constants/__init__.py)

The application loads environment variables via `python-dotenv` and exposes the following constants:

| Constant | Type | Description | Default/Source |
|----------|------|-------------|----------------|
| `GOOGLE_API_KEY` | str | Google Gemini API key | `os.getenv("GOOGLE_API_KEY")` |
| `EMBEDDING_MODEL` | str | Gemini embedding model name | `"models/text-embedding-004"` |
| `MODEL` | str | Gemini chat/generation model | `"gemini-2.5-flash"` |
| `CHROMA_API_KEY` | str | ChromaDB Cloud API key | `os.getenv("CHROMA_API_KEY")` |
| `CHROMA_TENANT` | str | ChromaDB Cloud tenant ID | `os.getenv("CHROMA_TENANT")` |
| `CHROMA_DATABASE` | str | ChromaDB Cloud database name | `os.getenv("CHROMA_DATABASE")` |
| `CHROMA_SERVER_HTTP_PORT` | int | ChromaDB server port | `443` (HTTPS) |
| `CHROMA_SERVER_HOST` | str | ChromaDB server host | `"api.trychroma.com"` |
| `COLLECTION_NAME` | str | Main vector collection name | `"insurance_data_mr_helpmate"` |
| `CACHE_COLLECTION_NAME` | str | Query cache collection name | `"query_insurance_data_mr_helpmate"` |
| `DISTANCE_THRESHOLD` | float | Cache similarity threshold | `0.1` (cosine distance) |

**Model Details:**
- **Embedding Model:** `text-embedding-004` produces 768-dimensional embeddings
- **Chat Model:** `gemini-2.5-flash` is optimized for low latency and high throughput

**Chunking Parameters (configurable in code):**
- Default `chunk_size`: 1200 characters
- Default `chunk_overlap`: 250 characters

**Retrieval Parameters (configurable in code):**
- Default `top_k`: 10 documents from vector search
- Default reranking `top_n`: 3 documents after cross-encoder

**Cross-Encoder Model:**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (loaded by `Reranker` class)

---

## Workflow 1: Embeddings Ingestion Pipeline

This workflow processes PDF documents and stores their embeddings in ChromaDB Cloud for semantic search.

### Pipeline Steps

**1. Read PDF Content**

Module: `src/backend/pdf_retriever.py`  
Class: `Extractor`  
Method: `content_extractor(file_path: str) -> str`

Extracts text and tables from a PDF file using `pdfplumber`. Returns a single concatenated string of all pages and tables separated by double newlines.

```python
from src.backend.pdf_retriever import Extractor

pdf_extractor = Extractor()
full_text = pdf_extractor.content_extractor("documents/sample-policy.pdf")
```

**2. Chunk the Text**

Module: `src/backend/chunking_layer.py`  
Class: `TextChunking`  
Method: `create_chunks(input_text: str, chunk_size: int = 1200, overlap: int = 250) -> list[str]`

Splits the PDF text into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`. Overlap ensures semantic continuity across chunk boundaries.

```python
from src.backend.chunking_layer import TextChunking

chunker = TextChunking()
chunks = chunker.create_chunks(full_text, chunk_size=1200, overlap=250)
```

**3. Generate Embeddings per Chunk**

Module: `src/database/chromadb_vectorstore.py`  
Class: `VectorStore`  
Method: `upsert_documents(documents: List[str])`

For each chunk, calls `genai.embed_content(model=EMBEDDING_MODEL, content=chunk)` to generate a 768-dimensional embedding vector. Embeddings are computed via the Google Gemini API.

**4. Upsert to ChromaDB**

Module: `src/database/chromadb_vectorstore.py`  
Class: `VectorStore`  

Connects to ChromaDB Cloud using credentials from `src/constants/` and upserts document chunks with:
- **documents**: List of text chunks
- **embeddings**: List of embedding vectors
- **ids**: Deterministic IDs generated via `generate_document_id(doc, index)` (MD5 hash-based)
- **metadatas**: Optional key-value metadata (currently not used but supported)

The `VectorStore` class uses `ChromaConnection` (from `src/database/chromadb_connection.py`) to initialize a ChromaDB Cloud client with `chromadb.CloudClient`.

```python
from src.database.chromadb_vectorstore import VectorStore

vector_store = VectorStore()  # Uses default collection from constants
vector_store.upsert_documents(documents=chunks)
```

### Complete Ingestion Example

```python
from src.backend.pdf_retriever import Extractor
from src.backend.chunking_layer import TextChunking
from src.database.chromadb_vectorstore import VectorStore

# Step 1: Extract text from PDF
pdf_extractor = Extractor()
full_text = pdf_extractor.content_extractor("documents/Principal-Sample-Life-Insurance-Policy.pdf")

# Step 2: Chunk the text
chunker = TextChunking()
chunks = chunker.create_chunks(full_text, chunk_size=1200, overlap=250)

# Step 3 & 4: Generate embeddings and upsert to ChromaDB
vector_store = VectorStore()
vector_store.upsert_documents(documents=chunks)

print(f"Ingested {len(chunks)} chunks into ChromaDB Cloud.")
```

### Alternative: Local ChromaDB (src/backend/embedding_layer.py)

For local development/testing, the `VectorEmbedding` class provides a similar interface with persistent or in-memory ChromaDB:

```python
from src.backend.embedding_layer import VectorEmbedding

vec_emb = VectorEmbedding(persistence=True, persist_directory="chromadb_collections")
collection = vec_emb.generate_embedding(
    collection_name="local_test",
    documents=chunks,
    ids=None  # Auto-generated if not provided
)
```

**Note:** The production system uses `VectorStore` (ChromaDB Cloud) for scalability.

---

## Workflow 2: RAG Chatbot Pipeline

The RAG chatbot implements a multi-stage pipeline to ensure safe, relevant, and accurate responses.

### Pipeline Steps

**1. Moderation Check**

Module: `src/backend/intent_moderation_check.py`  
Class: `ModerationCheck`  
Method: `check_moderation(input_message: str) -> bool`

Uses the Gemini model with a safety prompt (from `src/artifacts/__init__.py::ModerationCheckPrompt`) to classify user input as safe or unsafe. Returns `True` if content is flagged (unsafe), `False` if safe.

**Policy:** Flags hate speech, violence, harassment, self-harm, crime facilitation, extremism, PII leakage, harmful medical/financial advice, graphic content, and malware.

```python
from src.backend.intent_moderation_check import ModerationCheck

mod_check = ModerationCheck()
is_unsafe = mod_check.check_moderation(user_query)
if is_unsafe:
    print("Your message violates our content policy.")
    # Terminate conversation
```

**2. Intent Check**

Module: `src/backend/intent_moderation_check.py`  
Class: `IntentCheck`  
Method: `check_intent(input_message: str) -> bool`

Uses the Gemini model with an intent prompt (from `src/artifacts/__init__.py::IntentConfirmationPrompt`) to determine if the query is related to insurance, banking, or finance. Returns `True` if in-domain, `False` otherwise.

```python
from src.backend.intent_moderation_check import IntentCheck

intent_check = IntentCheck()
is_in_domain = intent_check.check_intent(user_query)
if not is_in_domain:
    print("I can only answer questions about insurance, banking, and finance.")
    # Terminate conversation
```

**3. Cache Lookup**

Module: `src/database/chromadb_cache.py`  
Class: `CacheVectorStore`  
Method: `check_query_in_cache(query_emb: List[float], threshold: float) -> Tuple[bool, Dict]`

Generates an embedding for the user query and searches the cache collection for similar queries (cosine distance < `DISTANCE_THRESHOLD`). If a cache hit is found, retrieves the stored metadata containing document IDs from the main collection.

**Cache Hit:**  
Fetch documents by IDs from the main collection (`VectorStore.collection.get(ids=...)`).

**Cache Miss:**  
Proceed to ChromaDB query (Step 4).

```python
from src.database.chromadb_vectorstore import VectorStore
from google import generativeai as genai
from src.constants import EMBEDDING_MODEL

vector_store = VectorStore()
query_embedding = genai.embed_content(model=EMBEDDING_MODEL, content=user_query)["embedding"]

# Cache check is internal to query_from_db
result = vector_store.query_from_db(query=user_query, top_k=10)
```

**4. Query ChromaDB**

Module: `src/database/chromadb_vectorstore.py`  
Class: `VectorStore`  
Method: `query_from_db(query: str, top_k: int = 10) -> Dict`

If cache miss, queries the main ChromaDB collection with the query embedding to retrieve the top_k most similar documents. Results include documents, embeddings, metadatas, distances, and IDs.

**5. Update Cache**

After a successful ChromaDB query, the top document IDs are stored in the cache collection with the query embedding:

```python
cache_vectorstore.update_cache(
    query=user_query,
    query_emb=query_embedding,
    metadata={"ids_json": json.dumps(top_ids)}
)
```

**6. Rerank with Cross-Encoder**

Module: `src/backend/reranker.py`  
Class: `Reranker`  
Method: `rerank_documents(documents: list[str], embeddings: list[list[float]], query: str, cross_encoder: bool = True, top_k: int = 3) -> list[list[float, str]]`

Uses a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to compute relevance scores for each (query, document) pair. Returns the top_k documents with the highest scores.

**Why Cross-Encoder?**  
Cross-encoders jointly encode query and document, providing more accurate relevance scores than cosine similarity alone.

```python
from src.backend.reranker import Reranker

reranker = Reranker()
top_docs = reranker.rerank_documents(
    documents=result['documents'],
    embeddings=result.get('embeddings', []),
    query=user_query,
    cross_encoder=True,
    top_k=3
)
# top_docs: [[score1, doc1], [score2, doc2], [score3, doc3]]
```

**7. Build System Context and Generate Response**

Module: `src/backend/generation_layer.py`  
Class: `ChatCompletion`  
Method: `chat_completion(messages: list[Dict[str, str]]) -> str`

Constructs a system message with the `SystemInstruction.prompt` from `src/artifacts/__init__.py`, which includes strict grounding rules. The context is built from reranked documents and passed to the Gemini chat model.

```python
from src.backend.generation_layer import ChatCompletion
from src.artifacts import SystemInstruction

chat_completion = ChatCompletion()

# Build context from top reranked documents
context = "\n\n".join([doc for score, doc in top_docs])
user_message = f"Context:\n{context}\n\nQuestion: {user_query}"

messages = [
    {"role": "system", "content": SystemInstruction.prompt},
    {"role": "user", "content": user_message}
]

response = chat_completion.chat_completion(messages=messages)
print(response)
```

**8. Maintain Chat History**

For multi-turn conversations, maintain a `messages` list and append user and assistant messages:

```python
messages = [{"role": "system", "content": SystemInstruction.prompt}]

for turn in conversation_loop:
    user_input = input("User: ")
    
    # Moderation, intent, retrieval, reranking...
    
    context = "\n\n".join([doc for score, doc in top_docs])
    user_message = f"Context:\n{context}\n\nQuestion: {user_input}"
    
    messages.append({"role": "user", "content": user_message})
    
    response = chat_completion.chat_completion(messages=messages)
    messages.append({"role": "assistant", "content": response})
    
    print(f"Assistant: {response}")
```

### Complete RAG Pipeline Example

```python
from src.backend.generation_layer import ChatCompletion
from src.artifacts import SystemInstruction
from src.backend.intent_moderation_check import IntentCheck, ModerationCheck
from src.database.chromadb_vectorstore import VectorStore
from src.backend.reranker import Reranker

# Initialize components
mod_chk = ModerationCheck()
int_cnf = IntentCheck()
chat_completion = ChatCompletion()
vector_store = VectorStore()
reranker = Reranker()

# Initialize conversation history
messages = [
    {"role": "system", "content": SystemInstruction.prompt}
]

# Conversation loop
while True:
    user_query = input("User: ")
    
    if user_query.lower() in ["exit", "quit"]:
        break
    
    # Step 1: Moderation check
    if mod_chk.check_moderation(user_query):
        print("Assistant: Your message violates our content policy.")
        continue
    
    # Step 2: Intent check
    if not int_cnf.check_intent(user_query):
        print("Assistant: I can only answer questions about insurance, banking, and finance.")
        continue
    
    # Step 3-5: Cache lookup / ChromaDB query / Cache update
    result = vector_store.query_from_db(query=user_query, top_k=10)
    
    if not result or not result.get('documents'):
        print("Assistant: I couldn't find relevant information.")
        continue
    
    # Step 6: Rerank with cross-encoder
    top_docs = reranker.rerank_documents(
        documents=result['documents'][0] if isinstance(result['documents'][0], list) else result['documents'],
        embeddings=result.get('embeddings', [[]])[0] if result.get('embeddings') else [],
        query=user_query,
        cross_encoder=True,
        top_k=3
    )
    
    # Step 7: Build context and generate response
    context = "\n\n".join([doc for score, doc in top_docs])
    user_message = f"Context:\n{context}\n\nQuestion: {user_query}"
    
    messages.append({"role": "user", "content": user_message})
    
    response = chat_completion.chat_completion(messages=messages)
    
    messages.append({"role": "assistant", "content": response})
    
    print(f"Assistant: {response}")
```

---

## Usage Examples

### Example 1: End-to-End Ingestion CLI

A complete script to ingest a PDF document into ChromaDB:

```python
#!/usr/bin/env python
"""
ingestion_cli.py - Ingest PDF documents into ChromaDB Cloud
"""
import sys
from src.backend.pdf_retriever import Extractor
from src.backend.chunking_layer import TextChunking
from src.database.chromadb_vectorstore import VectorStore

def main(pdf_path: str):
    print(f"[1/4] Extracting text from PDF: {pdf_path}")
    extractor = Extractor()
    full_text = extractor.content_extractor(pdf_path)
    
    if not full_text:
        print("Error: No text extracted from PDF.")
        return
    
    print(f"[2/4] Chunking text (chunk_size=1200, overlap=250)")
    chunker = TextChunking()
    chunks = chunker.create_chunks(full_text, chunk_size=1200, overlap=250)
    
    print(f"[3/4] Generating embeddings for {len(chunks)} chunks")
    print(f"[4/4] Upserting to ChromaDB Cloud")
    vector_store = VectorStore()
    vector_store.upsert_documents(documents=chunks)
    
    print(f"✓ Successfully ingested {len(chunks)} chunks into ChromaDB.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingestion_cli.py <path-to-pdf>")
        sys.exit(1)
    
    main(sys.argv[1])
```

Run:
```bash
python ingestion_cli.py documents/Principal-Sample-Life-Insurance-Policy.pdf
```

### Example 2: Interactive RAG Client

A complete interactive chatbot script (based on the user's provided example):

```python
#!/usr/bin/env python
"""
rag_client.py - Interactive RAG chatbot with moderation, intent, cache, and reranking
"""
from src.backend.generation_layer import ChatCompletion
from src.artifacts import SystemInstruction
from src.backend.intent_moderation_check import IntentCheck, ModerationCheck
from src.database.chromadb_vectorstore import VectorStore
from src.backend.reranker import Reranker

def main():
    # Initialize components
    mod_chk = ModerationCheck()
    int_cnf = IntentCheck()
    chat_completion = ChatCompletion()
    vector_store = VectorStore()
    reranker = Reranker()

    # Initialize conversation with system instruction
    messages = [
        {"role": "system", "content": SystemInstruction.prompt}
    ]
    
    print("Mr. HelpMate AI - Insurance & Banking Assistant")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        user_query = input("You: ").strip()
        
        if not user_query:
            continue
        
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Moderation check
        if mod_chk.check_moderation(user_query):
            print("Assistant: Your message violates our content policy. Please rephrase.\n")
            continue
        
        # Intent check
        if not int_cnf.check_intent(user_query):
            print("Assistant: I can only answer questions about insurance, banking, and finance.\n")
            continue
        
        # Retrieve documents (with cache lookup)
        try:
            result = vector_store.query_from_db(query=user_query, top_k=10)
        except Exception as e:
            print(f"Assistant: An error occurred during retrieval: {e}\n")
            continue
        
        # Handle ChromaDB query result structure
        documents = result.get('documents', [])
        embeddings = result.get('embeddings', [])
        
        # ChromaDB returns nested lists for query results
        if documents and isinstance(documents[0], list):
            documents = documents[0]
        if embeddings and isinstance(embeddings[0], list):
            embeddings = embeddings[0]
        
        if not documents:
            print("Assistant: I couldn't find relevant information to answer your question.\n")
            continue
        
        # Rerank documents
        try:
            top_docs = reranker.rerank_documents(
                documents=documents,
                embeddings=embeddings,
                query=user_query,
                cross_encoder=True,
                top_k=3
            )
        except Exception as e:
            print(f"Assistant: An error occurred during reranking: {e}\n")
            continue
        
        # Build context from reranked documents
        context = "\n\n".join([doc for score, doc in top_docs])
        user_message = f"Context:\n{context}\n\nQuestion: {user_query}"
        
        # Append to conversation history
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        try:
            response = chat_completion.chat_completion(messages=messages)
        except Exception as e:
            print(f"Assistant: An error occurred during generation: {e}\n")
            continue
        
        # Append assistant response to history
        messages.append({"role": "assistant", "content": response})
        
        print(f"Assistant: {response}\n")

if __name__ == "__main__":
    main()
```

Run:
```bash
python rag_client.py
```

**Sample Interaction:**

```
Mr. HelpMate AI - Insurance & Banking Assistant
Type 'exit' or 'quit' to end the conversation.

You: What is the deductible on the Gold Health Plan?
Assistant: The deductible is $500 per member per year.

You: How do I create a monthly budget?
Assistant: I don't have an answer.

You: exit
Goodbye!
```

---

## Testing

### Existing Tests

The repository includes informal test scripts:

- **`test.py`**: Demonstrates PDF extraction, chunking, vector store querying, and reranking with cross-encoder.
- **`test2.py`**: Tests moderation and intent checks with sample inputs (safe and unsafe messages).
- **`testing.ipynb`**, **`submission.ipynb`**: Jupyter notebooks for interactive experimentation with LangGraph, cross-encoders, and embeddings.

### Running Tests

```bash
# Test ingestion pipeline
python test.py

# Test moderation and intent checks
python test2.py
```

### Suggested Unit Tests

For production readiness, consider adding formal unit tests:

**Test Structure (using `pytest`):**

```
tests/
├── test_chunking.py
├── test_embeddings.py
├── test_vectorstore.py
├── test_reranker.py
├── test_moderation_intent.py
└── test_generation.py
```

**Example Test Cases:**

**`tests/test_chunking.py`:**
```python
from src.backend.chunking_layer import TextChunking

def test_create_chunks():
    chunker = TextChunking()
    text = "A" * 2000  # 2000 chars
    chunks = chunker.create_chunks(text, chunk_size=1000, overlap=200)
    assert len(chunks) >= 2
    assert len(chunks[0]) <= 1000

def test_overlap_works():
    chunker = TextChunking()
    chunks = chunker.create_chunks("ABCDEFGHIJ" * 100, chunk_size=50, overlap=10)
    # Check that overlap exists
    for i in range(len(chunks) - 1):
        assert chunks[i][-10:] == chunks[i+1][:10]
```

**`tests/test_moderation_intent.py`:**
```python
from src.backend.intent_moderation_check import ModerationCheck, IntentCheck

def test_moderation_flags_unsafe():
    mod = ModerationCheck()
    unsafe_message = "I want to kill someone"
    assert mod.check_moderation(unsafe_message) == True

def test_moderation_allows_safe():
    mod = ModerationCheck()
    safe_message = "What is a deductible?"
    assert mod.check_moderation(safe_message) == False

def test_intent_accepts_insurance_query():
    intent = IntentCheck()
    query = "What is my premium?"
    assert intent.check_intent(query) == True

def test_intent_rejects_unrelated_query():
    intent = IntentCheck()
    query = "Who won the World Cup?"
    assert intent.check_intent(query) == False
```

**`tests/test_reranker.py`:**
```python
from src.backend.reranker import Reranker

def test_reranker_cross_encoder():
    reranker = Reranker()
    docs = ["Insurance covers health.", "Soccer is a sport.", "Banking involves money."]
    query = "What does insurance cover?"
    top_docs = reranker.rerank_documents(documents=docs, query=query, top_k=2, cross_encoder=True)
    assert len(top_docs) == 2
    # Most relevant doc should be first
    assert "Insurance" in top_docs[0][1]
```

Run tests with:
```bash
pytest tests/
```

---

## Troubleshooting

### Common Issues

**1. `ValueError: GOOGLE_API_KEY is not set`**

**Cause:** Missing or invalid `.env` file.

**Solution:**
- Ensure `.env` exists in the project root.
- Verify `GOOGLE_API_KEY=your_key_here` is set.
- Restart your Python session to reload environment variables.

**2. `Failed to initialize ChromaDB Cloud client`**

**Cause:** Missing ChromaDB credentials or incorrect tenant/database.

**Solution:**
- Verify `CHROMA_API_KEY`, `CHROMA_TENANT`, and `CHROMA_DATABASE` in `.env`.
- Check network connectivity to `api.trychroma.com`.
- Ensure your ChromaDB Cloud account is active and the tenant/database exists.

**3. `No documents returned from query`**

**Cause:** Empty or incorrect collection, or query embedding mismatch.

**Solution:**
- Verify documents were ingested: `python test.py` (check upsert step).
- Inspect collection in ChromaDB Cloud dashboard.
- Ensure `COLLECTION_NAME` in `src/constants/__init__.py` matches your ingestion collection.

**4. `Cross-encoder model fails to load`**

**Cause:** Missing or incompatible `sentence-transformers` installation.

**Solution:**
```bash
pip install --upgrade sentence-transformers transformers
```

**5. `Chat response is "I don't have an answer"`**

**Cause:** Query is out of domain, or retrieved documents don't contain the answer.

**Solution:**
- Check intent check is passing (`IntentCheck.check_intent(query)` returns `True`).
- Verify PDF content was ingested correctly.
- Inspect retrieved documents before reranking to confirm relevance.

### Logging

All logs are written to `logs/` directory with timestamped filenames (e.g., `2024-01-15_14-30-00.log`). Check logs for detailed error messages and execution traces.

```bash
tail -f logs/$(ls -t logs/ | head -1)
```

---

## Glossary

**RAG (Retrieval-Augmented Generation):** A technique combining information retrieval with language generation. Retrieves relevant documents and uses them as context for generating accurate, grounded responses.

**Embeddings:** Dense vector representations of text capturing semantic meaning. Similar texts have similar embeddings (measured by cosine similarity or distance).

**ChromaDB:** Open-source vector database optimized for embeddings and similarity search. Supports cloud-hosted and local deployments.

**Cross-Encoder:** A neural model that jointly encodes query and document pairs to compute relevance scores. More accurate than bi-encoders (separate query/document embeddings) but slower.

**Reranking:** A second-stage retrieval step that refines initial results (e.g., top_k=10) by rescoring with a cross-encoder to select the most relevant subset (e.g., top_n=3).

**Cache Hit:** When a similar query exists in the cache (distance < threshold), allowing retrieval from stored metadata instead of re-querying the vector database.

**Moderation Check:** A safety filter that classifies user input for harmful content (hate speech, violence, harassment, etc.) before processing.

**Intent Check:** A domain filter that determines if a user query is related to insurance, banking, or finance. Out-of-domain queries are rejected.

**System Instruction (Prompt):** A detailed prompt template that guides the LLM's behavior, enforcing strict grounding, domain constraints, and output format.

**Grounding:** Ensuring the LLM's responses are based solely on provided context (retrieved documents), preventing hallucination or fabrication.

**Gemini API:** Google's generative AI API providing embeddings (`text-embedding-004`) and chat/completion models (`gemini-2.5-flash`).

**Chunk Overlap:** The number of characters shared between consecutive chunks. Ensures semantic continuity and prevents information loss at chunk boundaries.

**cosine similarity:** A metric measuring the similarity between two vectors. Values range from -1 (opposite) to 1 (identical). Embeddings with high cosine similarity are semantically related.

**Deterministic IDs:** Document IDs generated via a hash function (MD5) of the content. Ensures idempotent upserts (same document always gets the same ID).

---

## Additional Resources

- **LangChain Documentation:** [https://python.langchain.com/](https://python.langchain.com/)
- **ChromaDB Documentation:** [https://docs.trychroma.com/](https://docs.trychroma.com/)
- **Google Gemini API:** [https://ai.google.dev/](https://ai.google.dev/)
- **Sentence Transformers (Cross-Encoders):** [https://www.sbert.net/](https://www.sbert.net/)

---

**Last Updated:** 2024 (based on repository state at time of documentation generation)

**Maintainers:** Project contributors (see GitHub repository)

**License:** See `LICENSE` file in the repository root.
