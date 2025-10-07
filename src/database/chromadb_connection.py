import os
import chromadb
from chromadb.config import Settings
from src.constants import CHROMA_DB_API_KEY, COLLECTION_NAME
from src.logging import Logger

logging = Logger()

class ChromaConnection:
    def __init__(self, cloud_host: str = "https://api.trychroma.com/", port: int = 443):
        """
        Initialize the ChromaConnection instance by creating a ChromaDB REST client.

        Args:
            cloud_host (str): The ChromaDB REST API host URL.
            port (int): The HTTP port used to connect to the ChromaDB service.

        Raises:
            ValueError: If the required CHROMA_DB_API_KEY environment variable is missing.
            Exception: If initializing the ChromaDB client fails for any other reason.
        """
        self.CHROMA_DB_API_KEY = CHROMA_DB_API_KEY
        if not self.CHROMA_DB_API_KEY:
            logging.error("CHROMA_DB_API_KEY is not set.")
            raise ValueError("CHROMA_DB_API_KEY is required.")
        try:
            self.client = chromadb.Client(Settings(
                chroma_api_impl='rest',
                chroma_server_host=cloud_host,
                chroma_server_http_port=port,
                chroma_api_key=self.CHROMA_DB_API_KEY    
            ))
            logging.info("ChromaDB Cloud connection established")
        except Exception as e:
            logging.error(f"Failed to establish ChromaDB Cloud connection: {e}")
            raise e

    def get_client(self):
        """
        Retrieve the active ChromaDB client instance.

        Returns:
            chromadb.Client: The initialized ChromaDB client.

        Raises:
            Exception: If accessing the client results in an unexpected error.
        """
        try:
            if not self.client:
                logging.warning("ChromaDB client is not initialized.")
            return self.client
        except Exception as e:
            logging.error(f"Error getting ChromaDB client: {e}")
            raise e

    def get_or_create_collection(self, collection_name: str = COLLECTION_NAME):
        """
        Retrieve an existing ChromaDB collection or create a new one if it does not exist.

        Args:
            collection_name (str): The name of the collection to retrieve or create.

        Returns:
            chromadb.Collection: The requested or newly created ChromaDB collection.

        Raises:
            Exception: If the collection cannot be fetched or created.
        """
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            logging.info(f"Collection '{collection_name}' retrieved or created.")
            return collection
        except Exception as e:
            logging.error(f"Error getting or creating collection '{collection_name}': {e}")
            raise e