from src.backend.embedding_layer import VectorEmbedding
from src.database.chromadb_connection import ChromaConnection
from src.database.chromadb_cache import CacheVectorStore
from google import generativeai as genai
from src.constants import (COLLECTION_NAME, 
                           EMBEDDING_MODEL,
                           GOOGLE_API_KEY)
from src.logging import Logger
import hashlib

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set.")
genai.configure(api_key=GOOGLE_API_KEY)

logging = Logger()

class VectorStore:
    """
    Stores vector embeddings in Chroma Cloud and allows retrieval
    """
    def __init__(self, collection_name: str = COLLECTION_NAME, embedding_model: str = EMBEDDING_MODEL):
        try:
            logging.info("Initializing VectorStore.")
            self.conn = ChromaConnection()
            self.collection = self.conn.get_or_create_collection(collection_name=collection_name)
            self.embedding_model = embedding_model
            logging.info("VectorStore initialized successfully.")
        except Exception as exc:
            logging.error(f"Failed to initialize VectorStore: {exc}")
            raise

    def add_documents(self, documents):
        """
        Generate embeddings and store documents in the vector collection
        """
        try:
            if not documents.get('content'):
                logging.warning("No documents provided to add_documents.")
                return

            logging.info(f"Generating embeddings for {len(documents['content'])} documents.")
            embeddings = [
                genai.embed_content(model=self.embedding_model, content=doc)["embedding"]
                for doc in documents['content']
            ]

            ids_ = [
                hashlib.md5((documents['content'][i][:10] + "_" + str(i)).encode()).hexdigest()
                for i in range(len(documents['content']))
            ]

            logging.info("Upserting documents into the collection.")
            self.collection.upsert(
                documents=documents['content'],
                embeddings=embeddings,
                ids=ids_
            )
            logging.info("Documents added successfully.")
        except Exception as exc:
            logging.error(f"Failed to add documents: {exc}")
            raise

    def query_from_db(self, query: str = "", top_k: int = 10) -> list[str]:
        """
        Query vector store using embeddings of the query
        """
        try:
            if not query:
                logging.warning("Empty query received.")
                return []

            logging.info(f"Querying content with top_k={top_k}.")
            query_embedding = genai.embed_content(model=self.embedding_model, content=query)["embedding"]
            
            cache_vectorstore = CacheVectorStore()
            
            cache_status, ids = cache_vectorstore.check_query_in_cache(query_embedding)
            
            if cache_status:
                result = self.collection.get(
                    id=ids,
                    include=["documents", "metadatas"]
                )
                return result
            
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances", "ids"]
            )
            logging.info("Query completed successfully.")
            
            cache_vectorstore.update_cache(query=query,
                                           query_emb=query_embedding,
                                           metadata=result["metadatas"])
            return result
        except Exception as exc:
            logging.error(f"Failed to query content: {exc}")
            raise