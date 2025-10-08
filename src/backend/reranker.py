import heapq 

from sentence_transformers import CrossEncoder
from src.logging import Logger
import google.generativeai as genai
from src.constants import (GOOGLE_API_KEY, 
                           EMBEDDING_MODEL)
from sklearn.metrics.pairwise import cosine_similarity

logging = Logger()
genai.configure(api_key=GOOGLE_API_KEY)

class Reranker():
    def __init__(self):
        try:
            logging.info("Initializing Reranker with cross-encoder model")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            logging.error(f"Failed to initialize Reranker: {str(e)}")
            raise
    
    def rerank_documents(self, 
                         documents: list[str] = None,
                         embeddings: list[list[float]] = None,
                         query: str = None,
                         cross_encoder: bool = True,
                         top_k: int = 3) -> list[list[float, str]]:
        """
        Rerank documents using either cross-encoder or cosine similarity method.
        Args:
            documents: List of document strings
            embeddings: List of document embeddings
            query: Query string
            cross_encoder: Whether to use cross-encoder for reranking
            top_k: Number of top documents to return
        Returns:
            List of reranked documents
        """
        try:
            logging.info(f"Reranking documents with {'cross-encoder' if cross_encoder else 'cosine similarity'}")
            
            if cross_encoder:
                if documents is None or query is None:
                    logging.warning("Documents or query is None for cross-encoder reranking")
                    return []
                return self.rerank_with_cross_encoders(documents=documents, query=query, top_k=top_k)
            
            if documents is None or embeddings is None or query is None:
                logging.warning("Documents, embeddings, or query is None for cosine similarity reranking")
                return []
            return self.rerank_with_cosine_similarity(documents=documents, embeddings=embeddings, query=query, top_k=top_k)
            
        except Exception as e:
            logging.error(f"Error in rerank_documents: {str(e)}")
            raise
        
    def rerank_with_cross_encoders(self, 
                                   documents: list[str] = None,
                                   query: str = None,
                                   top_k: int = 3) -> list[list[float, str]]:
        """
        Rerank documents using cross-encoder model.
        Args:
            documents: List of document strings
            query: Query string
            top_k: Number of top documents to return
        Returns:
            List of [score, document] pairs sorted by score
        """
        try:
            logging.info(f"Reranking {len(documents)} documents with cross-encoder")
            hq_document = []
            for doc in documents:
                score = self.cross_encoder.predict([query, doc])
                heapq.heappush(hq_document, (score, doc))
                if len(hq_document) > top_k:
                    heapq.heappop(hq_document)

            top_docs = sorted(hq_document, key=lambda x: x[0], reverse=True)
            logging.info(f"Successfully reranked documents with cross-encoder")
            return [[score, doc] for score, doc in top_docs]
        except Exception as e:
            logging.error(f"Error in rerank_with_cross_encoders: {str(e)}")
            raise
    
    def rerank_with_cosine_similarity(self,
                                      documents: list[str] = None, 
                                      embeddings: list[list[float]] = None,
                                      query: str = None,
                                      top_k: int = 3) -> list[list[float, str]]:
        """
        Rerank documents using cosine similarity.
        Args:
            documents: List of document strings
            embeddings: List of document embeddings
            query: Query string
            top_k: Number of top documents to return
        Returns:
            List of [score, document] pairs sorted by score
        """
        try:
            if documents is None or embeddings is None or query is None:
                logging.warning("Missing required inputs for cosine similarity reranking")
                return []
                
            logging.info(f"Reranking {len(documents)} documents with cosine similarity")
            query_embedding_response = genai.embed_content(model=EMBEDDING_MODEL, content=query)
            query_embedding = query_embedding_response.get("embedding", query_embedding_response)
            
            hq_document = []
            for emb, doc in zip(embeddings, documents):
                score = cosine_similarity([query_embedding], [emb])[0][0]
                heapq.heappush(hq_document, (score, doc))
                if len(hq_document) > top_k:
                    heapq.heappop(hq_document)

            top_docs = sorted(hq_document, key=lambda x: x[0], reverse=True)
            logging.info(f"Successfully reranked documents with cosine similarity")
            return [[score, doc] for score, doc in top_docs]
        except Exception as e:
            logging.error(f"Error in rerank_with_cosine_similarity: {str(e)}")
            raise