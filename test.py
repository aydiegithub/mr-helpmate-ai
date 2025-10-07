from google import generativeai as genai
from src.constants import GOOGLE_API_KEY
from src.backend.embedding_layer import VectorEmbedding
from src.backend.pdf_retriever import Extractor

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Recommended embedding model (update from legacy "models/embedding-001")
EMBEDDING_MODEL = "models/text-embedding-004"

query = (
    "What is the Scheduled Benefit for all members under the Member Accidental Death and Dismemberment Insurance, "
    "and how can the approved amount differ from the Scheduled Benefit?"
)

# Get query embedding (use the full string, not query[0])
query_embedding = genai.embed_content(
    model=EMBEDDING_MODEL,
    content=query
)["embedding"]

# Initialize retriever and vector store
extractor = Extractor()
vec_emb = VectorEmbedding(persist_directory="chromadb_collection")

# Extract PDF content and index
content = extractor.content_extractor(r"documents/Principal-Sample-Life-Insurance-Policy.pdf")
documents = vec_emb.text_chunking(input_text=content['content'])
vec_emb.generate_embedding(documents=documents)

# Get the collection (prefer existing attribute if available; otherwise create/fetch a default)
collection = getattr(vec_emb, "collection", None)
if collection is None:
    # Try to reuse a known collection name on the wrapper, else fall back to "default"
    collection_name = getattr(vec_emb, "collection_name", "default")
    # get_or_create ensures it exists even if this script is run fresh
    collection = vec_emb.chroma_client.get_or_create_collection(name=collection_name)

# Query the vector store using the same embedding space
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=["documents", "metadatas", "distances"]  # "ids" is returned separately and not a valid include key
)

print(results)