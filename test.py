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

<<<<<<< HEAD
print(results)
=======
# ---- Neat Output for Retrieved Results ----
print("\n🔍 Top Retrieved Results")
print("────────────────────────────\n")

# Unpack results safely
ids_list = results.get("ids", [[]])[0]
distances = results.get("distances", [[]])[0]
documents = results.get("documents", [[]])[0]
metadatas = results.get("metadatas", [[]])[0]

# Print each result clearly
for idx, (doc_id, distance, document, metadata) in enumerate(zip(ids_list, distances, documents, metadatas), start=1):
    print(f"📄 Result #{idx}")
    print(f"   🆔 ID: {doc_id}")
    print(f"   📏 Distance: {distance:.6f}")
    print("   🧾 Document Snippet:")
    print(f"      {document.strip()[:500]}{'...' if len(document) > 500 else ''}")
    
    if metadata:
        print("   🗂️ Metadata:")
        for key, value in metadata.items():
            print(f"      - {key}: {value}")
    else:
        print("   🗂️ Metadata: None")
    
    print("────────────────────────────\n")

# Optional: Print summary
print(f"✅ Total Results Retrieved: {len(ids_list)}\n")
>>>>>>> chromadb-setup
