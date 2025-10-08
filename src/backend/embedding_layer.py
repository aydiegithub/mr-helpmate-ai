from src.logging import Logger
import chromadb
import chromadb.config
import os
from google import generativeai as genai
from src.constants import GOOGLE_API_KEY, EMBEDDING_MODEL
import hashlib


logging = Logger()
genai.configure(api_key=GOOGLE_API_KEY)
chromadb_collection = chromadb.api.models.Collection.Collection

class VectorEmbedding():
    """
    Handles embedding generation and persistence using ChromaDB and Gemini API.
    Supports both persistent and in-memory ChromaDB clients.
    """

    def __init__(self, persistence: bool = True, persist_directory: str = "chromadb_collections"):
        """
        Initializes the VectorEmbedding class with a ChromaDB client.
        If persistence is True, uses persistent storage at 'chromadb_collections'.
        Otherwise, uses an in-memory client without persistence.

        Args:
            persistence (bool, optional): Whether to persist data to disk. Defaults to True.
        """
        try:
            if persistence:
                logging.info(f"Initializing ChromaDB client with persistent storage at {persist_directory}.")
                self.chroma_client = chromadb.Client(chromadb.config.Settings(
                    persist_directory=persist_directory
                ))
                logging.info("ChromaDB client initialized successfully with persistence.")
            else:
                logging.info("Initializing ChromaDB client in-memory without persistence.")
                self.chroma_client = chromadb.Client()  # <- FIXED
                logging.info("ChromaDB client initialized successfully in-memory.")
        except Exception as e:
            logging.error(f"Error initializing ChromaDB client: {e}")
            raise

    def generate_embedding(
        self,
        collection_name: str = 'default',
        documents: list[str] | None = None,
        ids: list[str] | None = None
    ) -> chromadb_collection:
        """
        Generates embeddings for the provided documents using Gemini API and upserts them into ChromaDB.

        Args:
            collection_name (str, optional): The name of the ChromaDB collection. Defaults to 'default'.
            documents (list[str], optional): List of documents to embed. Defaults to [''].
            ids (list[str], optional): List of document IDs. If not provided, generated automatically.

        Returns:
            chromadb.api.models.Collection.Collection: The ChromaDB collection with the upserted documents.
        """
        try:
            if not documents:
                documents = ['']

            logging.info(f"Getting or creating ChromaDB collection: {collection_name}")
            collection = self.chroma_client.get_or_create_collection(name=collection_name)

            embeddings = []
            for idx, doc in enumerate(documents):
                try:
                    logging.info(f"Generating embedding for document {idx}")
                    emb_response = genai.embed_content(model=EMBEDDING_MODEL, content=doc)
                    embeddings.append(emb_response["embedding"])
                except Exception as e:
                    logging.error(f"Embedding generation failed for document {idx}: {e}")
                    embeddings.append([])
                    
            if not ids:
                ids = [self.generate_document_id(doc=doc, index=id) for id, doc in enumerate(documents)]

            logging.info(f"Upserting {len(documents)} documents into collection '{collection_name}'")
            collection.upsert(
                documents=documents,
                embeddings=embeddings,
                ids=ids
            )

            logging.info("Embedding generation and upsert completed successfully.")
            return collection

        except Exception as e:
            logging.error(f"Error during embedding generation or upsert: {e}")
            raise

    def save_collection(self, db_path: str = "chromadb_collections"):
        """
        Persists the ChromaDB collections to disk.

        Args:
            db_path (str, optional): The directory path to save the collections. Defaults to "chromadb_collections".
        """
        try:
            logging.info(f"Saving collection to {db_path}")
            if not os.path.exists(db_path):
                os.makedirs(db_path)
                logging.info(f"Created directory {db_path} for collection persistence.")
            self.chroma_client.persist()
            logging.info(f"Collection saved successfully at {db_path}")
        except Exception as e:
            logging.error(f"Error saving collection: {e}")
            
    
    def delete_documents(self, 
                         id: list[str] = None, 
                         delete_all: bool = False, 
                         confirm_delete_all: bool = False):
        """
        Delete documents from the collection by ID or delete all documents.

        Args:
            id (list[str], optional): List of document IDs to delete.
            delete_all (bool, optional): If True, deletes all documents. Defaults to False.
            confirm_delete_all (bool, optional): Must be True to confirm delete_all. Defaults to False.
        """
        try:
            if delete_all:
                if not confirm_delete_all:
                    logging.warning("confirm_delete_all=False. Set it to True to delete all documents.")
                    return
                logging.info("Deleting all documents from all collections.")
                for collection in self.chroma_client.list_collections():
                    coll_obj = self.chroma_client.get_collection(collection.name)
                    coll_obj.delete()
                logging.info("All documents deleted successfully.")
                return

            if id:
                logging.info(f"Deleting documents with IDs: {id}")
                for collection in self.chroma_client.list_collections():
                    coll_obj = self.chroma_client.get_collection(collection.name)
                    coll_obj.delete(ids=id)
                logging.info(f"Documents {id} deleted successfully.")
                return

            logging.warning("No ID provided and delete_all=False. Nothing was deleted.")

        except Exception as exc:
            logging.error(f"Failed to delete documents: {exc}")
            raise
            
    def generate_document_id(self, doc: str, index: int = 1) -> str:
        """
        Generate deterministic numeric ID with suffix n1, n2, etc.
        
        Args:
            doc (str): The document content
            index (int): Index for uniqueness (default 1)
        
        Returns:
            str: Generated ID, e.g., 748837283n1
        """
        doc_bytes = doc.encode("utf-8")
        
        hash_object = hashlib.md5(doc_bytes)
        
        hash_int = int(hash_object.hexdigest(), 16)
        hash_str = str(hash_int)[:9]
        
        generated_id = f"{hash_str}n{index}"
        
        return generated_id