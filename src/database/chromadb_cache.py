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

    def check_query_in_cache(
        self,
        query_emb: List[float],
        threshold: float = DISTANCE_THRESHOLD,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Checks if a similar query embedding exists in the cache.

        Args:
            query_emb (List[float]): Embedding vector of the query.
            threshold (float): Maximum cosine distance to consider a query as similar.

        Returns:
            Tuple[bool, dict]: (True, metadata) if a similar query exists, else (False, {}).
        """
        logging.info("Checking if query exists in cache.")
        if not query_emb:
            logging.info("Empty query embedding supplied to cache lookup.")
            return False, {}

        if threshold <= 0:
            logging.info(
                "Non-positive threshold provided (%s); using default (%s).",
                threshold,
                DISTANCE_THRESHOLD,
            )
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
                    "Closest cached query distance (%s) exceeds threshold (%s).",
                    top_distance,
                    threshold,
                )
                return False, {}

            metadata: Dict[str, Any] = {}
            if metadatas and metadatas[0]:
                raw_metadata = metadatas[0][0]
                if isinstance(raw_metadata, dict):
                    metadata = raw_metadata
                else:
                    logging.error(
                        "Unexpected metadata type from Chroma: %s", type(raw_metadata)
                    )
                    metadata = {"value": raw_metadata}

            logging.info("Cache hit found within threshold: %s", top_distance)
            return True, metadata

        except Exception as exc:
            logging.error(f"Error checking query in cache: {exc}")
            return False, {}

    def update_cache(
        self,
        query: str,
        query_emb: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Updates the Chroma cache with a new query, embedding, and metadata.

        Args:
            query (str): The FAQ query text.
            query_emb (List[float]): Embedding vector of the query.
            metadata (Dict[str, Any]): Metadata dictionary, e.g., {"indexes": [...]}
        """
        if not query or not query_emb or not metadata:
            logging.warning("Empty query, embedding, or metadata; skipping cache update.")
            return
        try:
            self.collection.upsert(
                documents=[query],
                embeddings=[query_emb],
                metadatas=[metadata],
            )
            logging.info("Cache updated successfully for query.")
        except Exception as exc:
            logging.error(f"Error updating cache: {exc}")