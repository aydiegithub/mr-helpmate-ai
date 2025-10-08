from src.constants import (
    CHROMA_API_KEY,
    CHROMA_DATABASE,
    CHROMA_TENANT,
)
import chromadb
from src.logging import Logger

logging = Logger()

class ChromaConnection:
    def __init__(self):
        """
        Initialize the ChromaConnection instance for ChromaDB Cloud using the official CloudClient.
        All configuration is read from src.constants (which loads .env via dotenv).
        """
        try:
            # Validate required configuration from constants
            missing = []
            if not CHROMA_API_KEY:
                missing.append("CHROMA_API_KEY")
            if not CHROMA_TENANT:
                missing.append("CHROMA_TENANT")
            if not CHROMA_DATABASE:
                missing.append("CHROMA_DATABASE")

            if missing:
                for var in missing:
                    logging.error(f"{var} is not set in src.constants.")
                raise ValueError(
                    "Missing required Chroma Cloud configuration. "
                    f"Set these in your .env so src.constants can load them: {', '.join(missing)}"
                )

            logging.info("Using Chroma tenant and database.")

            # Use the official Cloud client (recommended by Chroma docs)
            self.client = chromadb.CloudClient(
                api_key=CHROMA_API_KEY,
                tenant=CHROMA_TENANT,
                database=CHROMA_DATABASE,
            )
            logging.info("ChromaDB Cloud client initialized successfully.")

            # Verify existing collections (optional)
            existing = [c.name for c in self.client.list_collections()]
            logging.info(f"Existing collections in Chroma Cloud: {existing}")

        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB Cloud client: {e}")
            raise

    def get_client(self):
        """Return the active ChromaDB client instance."""
        try:
            if not hasattr(self, "client") or self.client is None:
                logging.warning("ChromaDB client is not initialized.")
            return self.client
        except Exception as e:
            logging.error(f"Error accessing ChromaDB client: {e}")
            raise

    def get_or_create_collection(self, collection_name: str):
        """
        Retrieve or create a ChromaDB collection.
        """
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            logging.info(f"Collection '{collection_name}' retrieved or created.")
            return collection
        except Exception as e:
            logging.error(f"Error getting or creating collection '{collection_name}': {e}")
            raise














# ----- Old Working Code -----


# from src.constants import (
#     CHROMA_API_KEY,
#     CHROMA_DATABASE,
#     CHROMA_TENANT,
# )
# import chromadb
# from src.logging import Logger

# logging = Logger()

# class ChromaConnection:
#     def __init__(self):
#         """
#         Initialize the ChromaConnection instance for ChromaDB Cloud using the official CloudClient.
#         All configuration is read from src.constants (which loads .env via dotenv).
#         """
#         try:
#             # Validate required configuration from constants
#             missing = []
#             if not CHROMA_API_KEY:
#                 missing.append("CHROMA_API_KEY")
#             if not CHROMA_TENANT:
#                 missing.append("CHROMA_TENANT")
#             if not CHROMA_DATABASE:
#                 missing.append("CHROMA_DATABASE")

#             if missing:
#                 for var in missing:
#                     logging.error(f"{var} is not set in src.constants.")
#                 raise ValueError(
#                     "Missing required Chroma Cloud configuration. "
#                     f"Set these in your .env so src.constants can load them: {', '.join(missing)}"
#                 )

#             logging.info(f"Using Chroma tenant and database.")

#             # Use the official Cloud client (recommended by Chroma docs)
#             self.client = chromadb.CloudClient(
#                 api_key=CHROMA_API_KEY,
#                 tenant=CHROMA_TENANT,
#                 database=CHROMA_DATABASE,
#             )
#             logging.info("ChromaDB Cloud client initialized successfully.")

#             # Verify existing collections (optional)
#             existing = [c.name for c in self.client.list_collections()]
#             logging.info(f"Existing collections in Chroma Cloud: {existing}")

#         except Exception as e:
#             logging.error(f"Failed to initialize ChromaDB Cloud client: {e}")
#             raise

#     def get_client(self):
#         """Return the active ChromaDB client instance."""
#         try:
#             if not hasattr(self, "client") or self.client is None:
#                 logging.warning("ChromaDB client is not initialized.")
#             return self.client
#         except Exception as e:
#             logging.error(f"Error accessing ChromaDB client: {e}")
#             raise

#     def get_or_create_collection(self, collection_name: str):
#         """
#         Retrieve or create a ChromaDB collection.
#         """
#         try:
#             collection = self.client.get_or_create_collection(name=collection_name)
#             logging.info(f"Collection '{collection_name}' retrieved or created.")
#             return collection
#         except Exception as e:
#             logging.error(f"Error getting or creating collection '{collection_name}': {e}")
#             raise