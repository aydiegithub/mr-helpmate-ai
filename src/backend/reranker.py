import heapq
from typing import List, Tuple, Optional

import numpy as np
from sentence_transformers import CrossEncoder
from src.logging import Logger
import google.generativeai as genai
from src.constants import (GOOGLE_API_KEY,
                           EMBEDDING_MODEL)
from sklearn.metrics.pairwise import cosine_similarity

logging = Logger()
genai.configure(api_key=GOOGLE_API_KEY)

class Reranker:
    def __init__(self):
        try:
            logging.info("Initializing Reranker with cross-encoder model")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            logging.error(f"Failed to initialize Reranker: {str(e)}")
            raise

    @staticmethod
    def _flatten_documents(documents: Optional[List]) -> List[str]:
        """
        Flattens documents that may be returned as a nested list (e.g., from vector stores).
        Ensures all entries are strings and filters out Nones/empties.
        """
        if not documents:
            return []
        # If documents is like [['doc1', 'doc2', ...]] flatten it
        if isinstance(documents[0], list):
            flat = [d for sub in documents for d in sub if d]
        else:
            flat = [d for d in documents if d]
        # Ensure string type
        return [str(d) for d in flat]

    @staticmethod
    def _flatten_embeddings(embeddings: Optional[List]) -> List[List[float]]:
        """
        Flattens embeddings to match flattened documents shape.
        """
        if not embeddings:
            return []
        if isinstance(embeddings[0], list) and embeddings and embeddings and isinstance(embeddings[0][0], (list, np.ndarray, tuple)):
            # embeddings like [[emb1, emb2, ...]]
            flat = [e for sub in embeddings for e in sub]
        else:
            flat = embeddings
        # Ensure list[float]
        out = []
        for e in flat:
            if e is None:
                continue
            if isinstance(e, np.ndarray):
                out.append(e.astype(float).tolist())
            elif isinstance(e, (list, tuple)):
                out.append([float(v) for v in e])
            else:
                # Unknown type, skip
                continue
        return out

    def rerank_documents(self,
                         documents: Optional[List[str]] = None,
                         embeddings: Optional[List[List[float]]] = None,
                         query: Optional[str] = None,
                         cross_encoder: bool = True,
                         top_k: int = 3) -> List[Tuple[float, str]]:
        """
        Rerank documents using either cross-encoder or cosine similarity method.
        Args:
            documents: List (possibly nested) of document strings
            embeddings: List (possibly nested) of document embeddings
            query: Query string
            cross_encoder: Whether to use cross-encoder for reranking
            top_k: Number of top documents to return
        Returns:
            List of (score, document) pairs sorted by score descending
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
                                   documents: Optional[List[str]] = None,
                                   query: Optional[str] = None,
                                   top_k: int = 3) -> List[Tuple[float, str]]:
        """
        Rerank documents using cross-encoder model.
        Args:
            documents: List (possibly nested) of document strings
            query: Query string
            top_k: Number of top documents to return
        Returns:
            List of (score, document) pairs sorted by score
        """
        try:
            docs = self._flatten_documents(documents)
            if not docs:
                logging.warning("No documents provided for cross-encoder reranking")
                return []

            logging.info(f"Reranking {len(docs)} documents with cross-encoder")

            # Prepare pairs as (query, doc)
            pairs = [(query, d) for d in docs]
            scores = self.cross_encoder.predict(pairs)  # np.ndarray of shape (len(docs),)

            # Select top_k indices
            k = min(top_k, len(docs))
            if k <= 0:
                return []

            # Use argpartition for efficiency, then sort those top indices
            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

            result = [(float(scores[i]), docs[i]) for i in top_idx]
            logging.info("Successfully reranked documents with cross-encoder")
            return result

        except Exception as e:
            logging.error(f"Error in rerank_with_cross_encoders: {str(e)}")
            raise

    def rerank_with_cosine_similarity(self,
                                      documents: Optional[List[str]] = None,
                                      embeddings: Optional[List[List[float]]] = None,
                                      query: Optional[str] = None,
                                      top_k: int = 3) -> List[Tuple[float, str]]:
        """
        Rerank documents using cosine similarity.
        Args:
            documents: List (possibly nested) of document strings
            embeddings: List (possibly nested) of document embeddings
            query: Query string
            top_k: Number of top documents to return
        Returns:
            List of (score, document) pairs sorted by score
        """
        try:
            docs = self._flatten_documents(documents)
            embs = self._flatten_embeddings(embeddings)

            if not docs or not embs or query is None:
                logging.warning("Missing required inputs for cosine similarity reranking")
                return []

            if len(docs) != len(embs):
                logging.warning(f"Documents and embeddings length mismatch: {len(docs)} vs {len(embs)}")
                # Align to min length
                min_len = min(len(docs), len(embs))
                docs = docs[:min_len]
                embs = embs[:min_len]

            logging.info(f"Reranking {len(docs)} documents with cosine similarity")

            query_embedding_response = genai.embed_content(model=EMBEDDING_MODEL, content=query)
            query_embedding = query_embedding_response.get("embedding", query_embedding_response)

            # Compute scores
            scores = []
            for emb in embs:
                score = float(cosine_similarity([query_embedding], [emb])[0][0])
                scores.append(score)

            k = min(top_k, len(docs))
            if k <= 0:
                return []

            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(np.array(scores)[top_idx])[::-1]]

            result = [(float(scores[i]), docs[i]) for i in top_idx]
            logging.info("Successfully reranked documents with cosine similarity")
            return result

        except Exception as e:
            logging.error(f"Error in rerank_with_cosine_similarity: {str(e)}")
            raise