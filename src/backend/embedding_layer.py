from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.logging import Logger
import chromadb
import chromadb.config
import os
from google import generativeai as genai
from src.constants import GOOGLE_API_KEY, EMBEDDING_MODEL

logging = Logger()
genai.configure(api_key=GOOGLE_API_KEY)

class VectorEmbedding():
    """
    Handles text chunking, embedding generation, and persistence using ChromaDB and Gemini API.
    """

    def __init__(self, persist_directory: str):
        """
        Initializes the VectorEmbedding class with a persistent ChromaDB client.
        """
        try:
            logging.info("Initializing ChromaDB client with persistent storage.")
            self.chroma_client = chromadb.Client(chromadb.config.Settings(
                persist_directory=persist_directory
            ))
            logging.info("ChromaDB client initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing ChromaDB client: {e}")
            raise

    def text_chunking(self, input_text: str, 
                      chunk_size: int = 1200, 
                      overlap: int = 250) -> list[str]:
        """
        Splits the input text into chunks of specified size with overlap.

        Args:
            input_text (str): The text to be chunked.
            chunk_size (int, optional): The size of each chunk. Defaults to 1200.
            overlap (int, optional): The number of overlapping characters between chunks. Defaults to 250.

        Returns:
            list[str]: A list of text chunks.
        """
        try:
            logging.info(f"Starting text chunking with chunk_size={chunk_size}, overlap={overlap}")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )
            chunks = text_splitter.split_text(input_text)
            logging.info(f"Text chunking successful. Number of chunks created: {len(chunks)}")
            if not chunks:
                logging.warning("No chunks were created from the input text.")
            return chunks

        except Exception as e:
            logging.error(f"Error during text chunking: {e}")
            return []

    def generate_embedding(self, collection_name: str = 'default',
                           documents: list[str] = None):
        """
        Generates embeddings for the provided documents using Gemini API and upserts them into ChromaDB.

        Args:
            collection_name (str, optional): The name of the ChromaDB collection. Defaults to 'default'.
            documents (list[str], optional): List of documents to embed. Defaults to [''].
        """
        try:
            if documents is None:
                documents = ['']
            logging.info(f"Getting or creating ChromaDB collection: {collection_name}")
            collection = self.chroma_client.get_or_create_collection(name=collection_name)
            embeddings = []
            for idx, doc in enumerate(documents):
                logging.info(f"Generating embedding for document {idx}")
                emb_response = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=doc
                )
                embeddings.append(emb_response["embedding"])
            ids = [f"doc_{i}" for i in range(len(documents))]
            logging.info(f"Upserting {len(documents)} documents into collection '{collection_name}'")
            collection.upsert(
                documents=documents,
                embeddings=embeddings,
                ids=ids
            )
            logging.info("Embedding generation and upsert completed successfully.")
        except Exception as e:
            logging.error(f"Error during embedding generation or upsert: {e}")

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