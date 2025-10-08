from src.constants import (
    CACHE_COLLECTION_NAME,
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    DISTANCE_THRESHOLD,
)
from src.database.chromadb_connection import ChromaConnection
from google import generativeai as genai
from src.logging import Logger
from typing import Any, Dict, Tuple, List
import hashlib
import json

logging = Logger()
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set.")
genai.configure(api_key=GOOGLE_API_KEY)

class CacheVectorStore:
    """
    Stores query embeddings in Chroma Cloud as a universal FAQ cache and allows retrieval.
    Each cache entry contains:
        - the query text
        - the query embedding
        - metadata (usually indexes of the main DB embeddings)
    """

    def __init__(
        self,
        collection_name: str = CACHE_COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        try:
            logging.info("Initializing CacheVectorStore.")
            self.conn = ChromaConnection()
            self.collection = self.conn.get_or_create_collection(collection_name=collection_name)
            self.embedding_model = embedding_model
            logging.info("CacheVectorStore initialized successfully.")
        except Exception as exc:
            logging.error(f"Failed to initialize CacheVectorStore: {exc}")
            raise

    def _make_cache_id(self, query: str) -> str:
        """
        Create a stable ID for the cache entry based on the query text.
        """
        base = query.strip().lower().encode("utf-8")
        h = hashlib.md5(base).hexdigest()[:24]
        return f"q_{h}"

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chroma Cloud metadata must be primitives (str, int, float, bool, None) or SparseVector.
        Convert lists/dicts to JSON strings with a _json suffix to remain queryable.
        """
        sanitized: Dict[str, Any] = {}
        for k, v in (metadata or {}).items():
            if v is None or isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            elif isinstance(v, (list, dict)):
                key = f"{k}_json" if not k.endswith("_json") else k
                try:
                    sanitized[key] = json.dumps(v)
                except Exception:
                    # Fallback to string if for some reason json fails
                    sanitized[key] = str(v)
                # Do NOT keep the original non-primitive under the same key
            else:
                sanitized[k] = str(v)
        return sanitized

    def check_query_in_cache(
        self,
        query_emb: List[float],
        threshold: float = DISTANCE_THRESHOLD,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Checks if a similar query embedding exists in the cache.
        """
        logging.info("Checking if query exists in cache.")
        # Be explicit; avoid NumPy truthiness ambiguity
        if query_emb is None or (hasattr(query_emb, "__len__") and len(query_emb) == 0):
            logging.info("Empty query embedding supplied to cache lookup.")
            return False, {}

        if threshold <= 0:
            logging.info(f"Non-positive threshold provided {threshold}; using default {DISTANCE_THRESHOLD}.")
            threshold = DISTANCE_THRESHOLD

        try:
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=1,
                include=["metadatas", "distances"],
            )

            if not results:
                logging.info("Chroma query returned no results.")
                return False, {}

            distances = results.get("distances") or []
            metadatas = results.get("metadatas") or []

            if not distances or not distances[0]:
                logging.info("No distance data returned for query.")
                return False, {}

            top_distance = distances[0][0]
            if top_distance >= threshold:
                logging.info(
                    f"Closest cached query distance ({top_distance}) exceeds threshold ({threshold})."
                )
                return False, {}

            metadata: Dict[str, Any] = {}
            if metadatas and metadatas[0]:
                raw_metadata = metadatas[0][0]
                if isinstance(raw_metadata, dict):
                    metadata = raw_metadata
                else:
                    logging.error(f"Unexpected metadata type from Chroma: {type(raw_metadata)}")
                    metadata = {"value": raw_metadata}

            logging.info(f"Cache hit found within threshold: {top_distance}")
            return True, metadata

        except Exception as exc:
            logging.error(f"Error checking query in cache: {exc}")
            return False, {}

    def update_cache(
        self,
        query: str,
        query_emb: List[float],
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Updates the Chroma cache with a new query, embedding, and metadata.
        Returns True on success, False otherwise.
        """
        # Be explicit; avoid NumPy truthiness ambiguity
        if not query or query_emb is None or (hasattr(query_emb, "__len__") and len(query_emb) == 0) or not metadata:
            logging.warning("Empty query, embedding, or metadata; skipping cache update.")
            return False
        try:
            cache_id = self._make_cache_id(query)
            sanitized_meta = self._sanitize_metadata(metadata)
            self.collection.upsert(
                ids=[cache_id],
                documents=[query],
                embeddings=[query_emb],
                metadatas=[sanitized_meta],
            )
            logging.info(f"Cache updated successfully for query. id={cache_id}")
            return True
        except Exception as exc:
            logging.error(f"Error updating cache: {exc}")
            return False



            
            
            
            







# ----- Old Working Code -----

# from src.constants import (
#     CACHE_COLLECTION_NAME,
#     GOOGLE_API_KEY,
#     EMBEDDING_MODEL,
#     DISTANCE_THRESHOLD,
# )
# from src.database.chromadb_connection import ChromaConnection
# from google import generativeai as genai
# from src.logging import Logger
# from typing import Any, Dict, Tuple, List

# logging = Logger()
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY is not set.")
# genai.configure(api_key=GOOGLE_API_KEY)

# class CacheVectorStore:
#     """
#     Stores query embeddings in Chroma Cloud as a universal FAQ cache and allows retrieval.
#     Each cache entry contains:
#         - the query text
#         - the query embedding
#         - metadata (usually indexes of the main DB embeddings)
#     """

#     def __init__(
#         self,
#         collection_name: str = CACHE_COLLECTION_NAME,
#         embedding_model: str = EMBEDDING_MODEL,
#     ):
#         try:
#             logging.info("Initializing CacheVectorStore.")
#             self.conn = ChromaConnection()
#             self.collection = self.conn.get_or_create_collection(collection_name=collection_name)
#             self.embedding_model = embedding_model
#             logging.info("CacheVectorStore initialized successfully.")
#         except Exception as exc:
#             logging.error(f"Failed to initialize CacheVectorStore: {exc}")
#             raise

#     def check_query_in_cache(
#         self,
#         query_emb: List[float],
#         threshold: float = DISTANCE_THRESHOLD,
#     ) -> Tuple[bool, Dict[str, Any]]:
#         """
#         Checks if a similar query embedding exists in the cache.
#         """
#         logging.info("Checking if query exists in cache.")
#         # Be explicit; avoid NumPy truthiness ambiguity
#         if query_emb is None or (hasattr(query_emb, "__len__") and len(query_emb) == 0):
#             logging.info("Empty query embedding supplied to cache lookup.")
#             return False, {}

#         if threshold <= 0:
#             logging.info(
#                 "Non-positive threshold provided (%s); using default (%s).",
#                 threshold,
#                 DISTANCE_THRESHOLD,
#             )
#             threshold = DISTANCE_THRESHOLD

#         try:
#             results = self.collection.query(
#                 query_embeddings=[query_emb],
#                 n_results=1,
#                 include=["metadatas", "distances"],
#             )

#             if not results:
#                 logging.info("Chroma query returned no results.")
#                 return False, {}

#             distances = results.get("distances") or []
#             metadatas = results.get("metadatas") or []

#             if not distances or not distances[0]:
#                 logging.info("No distance data returned for query.")
#                 return False, {}

#             top_distance = distances[0][0]
#             if top_distance >= threshold:
#                 logging.info(
#                     "Closest cached query distance (%s) exceeds threshold (%s).",
#                     top_distance,
#                     threshold,
#                 )
#                 return False, {}

#             metadata: Dict[str, Any] = {}
#             if metadatas and metadatas[0]:
#                 raw_metadata = metadatas[0][0]
#                 if isinstance(raw_metadata, dict):
#                     metadata = raw_metadata
#                 else:
#                     logging.error(
#                         "Unexpected metadata type from Chroma: %s", type(raw_metadata)
#                     )
#                     metadata = {"value": raw_metadata}

#             logging.info("Cache hit found within threshold: %s", top_distance)
#             return True, metadata

#         except Exception as exc:
#             logging.error(f"Error checking query in cache: {exc}")
#             return False, {}

#     def update_cache(
#         self,
#         query: str,
#         query_emb: List[float],
#         metadata: Dict[str, Any],
#     ) -> None:
#         """
#         Updates the Chroma cache with a new query, embedding, and metadata.
#         """
#         # Be explicit; avoid NumPy truthiness ambiguity
#         if not query or query_emb is None or (hasattr(query_emb, "__len__") and len(query_emb) == 0) or not metadata:
#             logging.warning("Empty query, embedding, or metadata; skipping cache update.")
#             return
#         try:
#             self.collection.upsert(
#                 documents=[query],
#                 embeddings=[query_emb],
#                 metadatas=[metadata],
#             )
#             logging.info("Cache updated successfully for query.")
#         except Exception as exc:
#             logging.error(f"Error updating cache: {exc}")