from src.backend.pdf_retriever import Extractor
from src.backend.chunking_layer import TextChunking
from src.backend.embedding_layer import VectorEmbedding  # noqa: F401 (kept for parity)
from src.database.chromadb_vectorstore import VectorStore

# Retrieve info from PDF
pdf_ret = Extractor()
docs = pdf_ret.content_extractor(r"documents/Principal-Sample-Life-Insurance-Policy.pdf")
print(docs[:100])
print(type(docs))

text_chunking = TextChunking()
chunks_main = text_chunking.create_chunks(docs)
chunks = chunks_main[50:53]

# Safely preview a couple of chunks if available
# if len(chunks) >= 2:
#     print("\n\nChunk1:\n\n", chunks[0], "\n\nChunk2:\n\n", chunks[1])
# print(len(chunks))

print("\n\n📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦📦")
# Pushing to chromadb cloud db
vector_store = VectorStore()

# Upsert documents so they are written
# vector_store.upsert_documents(chunks)

# Prefer fetching directly with explicit include to ensure embeddings are requested
# collection = vector_store.conn.get_or_create_collection(collection_name="insurance_data_mr_helpmate")

# print(type(collection))
# print(collection.get(include=["documents", "embeddings", "metadatas"])['embeddings'][0])

# vec_emb = VectorEmbedding()
# print(collection.query(vec_emb.generate_embedding("What is the possibility of insurence if someone dies")['embeddings']))

print(vector_store.query_from_db(query="What is the possibility of insurence in place of accident", 
                                 top_k=5))