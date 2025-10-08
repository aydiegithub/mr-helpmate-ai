from src.database.chromadb_connection import ChromaConnection
from src.database.chromadb_cache import CacheVectorStore
from google import generativeai as genai
from src.constants import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
)
from src.logging import Logger
import hashlib
import json
from typing import Dict, List, Any

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

    def upsert_documents(self, documents: List[str] = None):
        """
        Generate embeddings and store documents in the vector collection
        """
        try:
            if not documents:
                logging.warning("No documents provided to add_documents.")
                return

            logging.info(f"Generating embeddings for {len(documents)} documents.")
            embeddings = [
                genai.embed_content(model=self.embedding_model, content=doc)["embedding"]
                for doc in documents
            ]

            ids_ = [self.generate_document_id(doc=doc, index=i) for i, doc in enumerate(documents)]

            logging.info("Upserting documents into the collection.")
            self.collection.upsert(
                documents=documents,
                embeddings=embeddings,
                ids=ids_,
            )
            logging.info("Documents added successfully.")
        except Exception as exc:
            logging.error(f"Failed to add documents: {exc}")
            raise

    def fetch_all_data(self, collection_name: str = COLLECTION_NAME, batch_size: int = 500) -> Dict[str, Any]:
        """
        Fetch all documents, IDs, embeddings, and metadatas from Chroma DB collection.
        Uses ID pagination to ensure embeddings come back even if the server limits payload size.
        """
        try:
            logging.info(f"Fetching all data from collection {collection_name}.")
            collection = self.conn.get_or_create_collection(collection_name=collection_name)

            # 'ids' are returned by default; do not put 'ids' in include
            ids_resp = collection.get() or {}
            all_ids = ids_resp.get("ids") or []
            if not all_ids:
                logging.info(f"No records found in collection {collection_name}.")
                return {"ids": [], "documents": [], "embeddings": [], "metadatas": []}

            agg_ids: List[str] = []
            agg_docs: List[str] = []
            agg_embs: List[List[float]] = []
            agg_metas: List[Dict[str, Any]] = []

            for start in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[start:start + batch_size]
                batch = collection.get(
                    ids=batch_ids,
                    include=["documents", "embeddings", "metadatas"],
                ) or {}

                agg_ids.extend(batch.get("ids") or [])
                agg_docs.extend(batch.get("documents") or [])
                agg_embs.extend(batch.get("embeddings") or [])
                agg_metas.extend(batch.get("metadatas") or [])

            logging.info(
                f"Fetched {len(agg_docs)} documents, {len(agg_embs)} embeddings, {len(agg_metas)} metadatas from {collection_name}."
            )
            return {
                "ids": agg_ids,
                "documents": agg_docs,
                "embeddings": agg_embs,
                "metadatas": agg_metas,
            }
        except Exception as exc:
            logging.error(f"Failed to fetch all data from collection {collection_name}: {exc}")
            raise

    def _extract_ids_from_cached_meta(self, cached_meta: Dict[str, Any]) -> List[str]:
        """
        Supports multiple ways of storing IDs in cache metadata:
          - ids as list (if ever allowed)
          - ids_json as JSON string
          - ids_csv as comma-separated string
        """
        if not isinstance(cached_meta, dict):
            return []
        ids = cached_meta.get("ids")
        if isinstance(ids, list):
            return ids
        if "ids_json" in cached_meta and isinstance(cached_meta["ids_json"], str):
            try:
                parsed = json.loads(cached_meta["ids_json"])
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        if "ids_csv" in cached_meta and isinstance(cached_meta["ids_csv"], str):
            return [x for x in cached_meta["ids_csv"].split(",") if x]
        return []

    def query_from_db(self, query: str = "", top_k: int = 10) -> Dict[str, Any]:
        """
        Query vector store using embeddings of the query.
        Workflow:
          - Embed query
          - Cache check (distance < threshold => hit)
          - If hit: fetch by IDs from main collection
          - If miss: query main collection and cache top IDs
        """
        try:
            if not query:
                logging.warning("Empty query received.")
                return {}

            logging.info(f"Querying content with top_k={top_k}.")
            query_embedding = genai.embed_content(model=self.embedding_model, content=query)["embedding"]

            cache_vectorstore = CacheVectorStore()

            cache_status, cached_meta = cache_vectorstore.check_query_in_cache(query_embedding)

            if cache_status:
                logging.info(f"Query found in cache: {query}")
                ids = self._extract_ids_from_cached_meta(cached_meta)
                if ids:
                    # Fetch by IDs from the main collection
                    return self.collection.get(ids=ids, include=["documents", "metadatas"])

            logging.info("Query not found in cache.")
            # Do not include 'ids' here; Chroma returns 'ids' by default for query
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
            logging.info("Query completed successfully.")

            top_ids = result.get("ids", [[]])[0] if result.get("ids") else []
            if top_ids:
                # Store IDs under a JSON-serialized metadata key to satisfy Chroma type constraints
                success = cache_vectorstore.update_cache(
                    query=query,
                    query_emb=query_embedding,
                    metadata={"ids_json": json.dumps(top_ids)},
                )
                if success:
                    logging.info(f"Query cached with {len(top_ids)} ids.")
                else:
                    logging.info("Cache update failed; skipping success log.")
            else:
                logging.info("No IDs returned to cache; skipping cache update.")

            return result
        except Exception as exc:
            logging.error(f"Failed to query content: {exc}")
            raise

    def delete_documents(self, id: List[str] = None, delete_all: bool = False, confirm_delete_all: bool = False):
        """
        Delete documents from the collection by ID or delete all documents.
        """
        try:
            if delete_all:
                if not confirm_delete_all:
                    logging.warning("confirm_delete_all=False. Set it to True to delete all documents.")
                    return
                logging.info("Deleting all documents from the collection.")
                self.collection.delete()
                logging.info("All documents deleted successfully.")
                return

            if id:
                logging.info(f"Deleting documents with IDs: {id}")
                self.collection.delete(ids=id)
                logging.info(f"Documents {id} deleted successfully.")
                return

            logging.warning("No ID provided and delete_all=False. Nothing was deleted.")

        except Exception as exc:
            logging.error(f"Failed to delete documents: {exc}")
            raise

    def generate_document_id(self, doc: str, index: int = 1) -> str:
        """
        Generate deterministic string ID with suffix n1, n2, etc.
        """
        doc_bytes = doc.encode("utf-8")
        hash_object = hashlib.md5(doc_bytes)
        hash_int = int(hash_object.hexdigest(), 16)
        hash_str = str(hash_int)[:9]
        return f"{hash_str}n{index}"

















# ----- Old Working Code -----

# from src.database.chromadb_connection import ChromaConnection
# from src.database.chromadb_cache import CacheVectorStore
# from google import generativeai as genai
# from src.constants import (
#     COLLECTION_NAME,
#     EMBEDDING_MODEL,
#     GOOGLE_API_KEY,
# )
# from src.logging import Logger
# import hashlib
# from typing import Dict, List, Any

# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY is not set.")
# genai.configure(api_key=GOOGLE_API_KEY)

# logging = Logger()

# class VectorStore:
#     """
#     Stores vector embeddings in Chroma Cloud and allows retrieval
#     """
#     def __init__(self, collection_name: str = COLLECTION_NAME, embedding_model: str = EMBEDDING_MODEL):
#         try:
#             logging.info("Initializing VectorStore.")
#             self.conn = ChromaConnection()
#             self.collection = self.conn.get_or_create_collection(collection_name=collection_name)
#             self.embedding_model = embedding_model
#             logging.info("VectorStore initialized successfully.")
#         except Exception as exc:
#             logging.error(f"Failed to initialize VectorStore: {exc}")
#             raise

#     def upsert_documents(self, documents: List[str] = None):
#         """
#         Generate embeddings and store documents in the vector collection
#         """
#         try:
#             if not documents:
#                 logging.warning("No documents provided to add_documents.")
#                 return

#             logging.info(f"Generating embeddings for {len(documents)} documents.")
#             embeddings = [
#                 genai.embed_content(model=self.embedding_model, content=doc)["embedding"]
#                 for doc in documents
#             ]

#             ids_ = [self.generate_document_id(doc=doc, index=i) for i, doc in enumerate(documents)]

#             logging.info("Upserting documents into the collection.")
#             self.collection.upsert(
#                 documents=documents,
#                 embeddings=embeddings,
#                 ids=ids_,
#             )
#             logging.info("Documents added successfully.")
#         except Exception as exc:
#             logging.error(f"Failed to add documents: {exc}")
#             raise

#     def fetch_all_data(self, collection_name: str = COLLECTION_NAME, batch_size: int = 500) -> Dict[str, Any]:
#         """
#         Fetch all documents, IDs, embeddings, and metadatas from Chroma DB collection.
#         Uses ID pagination to ensure embeddings come back even if the server limits payload size.
#         """
#         try:
#             logging.info(f"Fetching all data from collection {collection_name}.")
#             collection = self.conn.get_or_create_collection(collection_name=collection_name)

#             ids_resp = collection.get(include=["ids"]) or {}
#             all_ids = ids_resp.get("ids") or []
#             if not all_ids:
#                 logging.info(f"No records found in collection {collection_name}.")
#                 return {"ids": [], "documents": [], "embeddings": [], "metadatas": []}

#             agg_ids: List[str] = []
#             agg_docs: List[str] = []
#             agg_embs: List[List[float]] = []
#             agg_metas: List[Dict[str, Any]] = []

#             for start in range(0, len(all_ids), batch_size):
#                 batch_ids = all_ids[start:start + batch_size]
#                 batch = collection.get(
#                     ids=batch_ids,
#                     include=["ids", "documents", "embeddings", "metadatas"],
#                 ) or {}

#                 agg_ids.extend(batch.get("ids") or [])
#                 agg_docs.extend(batch.get("documents") or [])
#                 agg_embs.extend(batch.get("embeddings") or [])
#                 agg_metas.extend(batch.get("metadatas") or [])

#             logging.info(
#                 f"Fetched {len(agg_docs)} documents, {len(agg_embs)} embeddings, {len(agg_metas)} metadatas from {collection_name}."
#             )
#             return {
#                 "ids": agg_ids,
#                 "documents": agg_docs,
#                 "embeddings": agg_embs,
#                 "metadatas": agg_metas,
#             }
#         except Exception as exc:
#             logging.error(f"Failed to fetch all data from collection {collection_name}: {exc}")
#             raise

#     def query_from_db(self, query: str = "", top_k: int = 10) -> Dict[str, Any]:
#         """
#         Query vector store using embeddings of the query.
#         Workflow:
#           - Embed query
#           - Cache check (distance < threshold => hit)
#           - If hit: fetch by IDs from main collection
#           - If miss: query main collection and cache top IDs
#         """
#         try:
#             if not query:
#                 logging.warning("Empty query received.")
#                 return {}

#             logging.info(f"Querying content with top_k={top_k}.")
#             query_embedding = genai.embed_content(model=self.embedding_model, content=query)["embedding"]

#             cache_vectorstore = CacheVectorStore()

#             cache_status, cached_meta = cache_vectorstore.check_query_in_cache(query_embedding)

#             if cache_status:
#                 logging.info(f"Query found in cache: {query}")
#                 ids = cached_meta.get("ids") if isinstance(cached_meta, dict) else None
#                 if ids:
#                     # Include 'ids' for uniform downstream handling
#                     return self.collection.get(ids=ids, include=["documents", "metadatas"])
#             logging.info(f"Query not found in cache.")
#             result = self.collection.query(
#                 query_embeddings=[query_embedding],
#                 n_results=top_k,
#                 include=["documents", "metadatas", "distances"],
#             )
#             logging.info("Query completed successfully.")

#             top_ids = result.get("ids", [[]])[0] if result.get("ids") else []
#             logging.info(f"Query updated to cache: {query}")
#             cache_vectorstore.update_cache(query=query, query_emb=query_embedding, metadata={"ids": top_ids})

#             return result
#         except Exception as exc:
#             logging.error(f"Failed to query content: {exc}")
#             raise

#     def delete_documents(self, id: List[str] = None, delete_all: bool = False, confirm_delete_all: bool = False):
#         """
#         Delete documents from the collection by ID or delete all documents.
#         """
#         try:
#             if delete_all:
#                 if not confirm_delete_all:
#                     logging.warning("confirm_delete_all=False. Set it to True to delete all documents.")
#                     return
#                 logging.info("Deleting all documents from the collection.")
#                 self.collection.delete()
#                 logging.info("All documents deleted successfully.")
#                 return

#             if id:
#                 logging.info(f"Deleting documents with IDs: {id}")
#                 self.collection.delete(ids=id)
#                 logging.info(f"Documents {id} deleted successfully.")
#                 return

#             logging.warning("No ID provided and delete_all=False. Nothing was deleted.")

#         except Exception as exc:
#             logging.error(f"Failed to delete documents: {exc}")
#             raise

#     def generate_document_id(self, doc: str, index: int = 1) -> str:
#         """
#         Generate deterministic string ID with suffix n1, n2, etc.
#         """
#         doc_bytes = doc.encode("utf-8")
#         hash_object = hashlib.md5(doc_bytes)
#         hash_int = int(hash_object.hexdigest(), 16)
#         hash_str = str(hash_int)[:9]
#         return f"{hash_str}n{index}"