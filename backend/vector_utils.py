from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json
import pickle
from typing import List, Dict, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = 384
        self.index_file = "backend/storage/faiss.index"
        self.metadata_file = "backend/storage/document_metadata.pkl"
        self.chunk_file = "backend/storage/text_chunks.json"

        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.chunks = []
        self.document_metadata = {}

        self._load_from_disk()

    def _load_from_disk(self):
        try:
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            if os.path.exists(self.chunk_file):
                with open(self.chunk_file, 'r') as f:
                    self.chunks = json.load(f)
                logger.info(f"Loaded {len(self.chunks)} text chunks")
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'rb') as f:
                    self.document_metadata = pickle.load(f)
                logger.info(f"Loaded metadata for {len(self.document_metadata)} documents")
        except Exception as e:
            logger.error(f"Error loading from disk: {str(e)}")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.chunks = []
            self.document_metadata = {}

    def _save_to_disk(self):
        try:
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            faiss.write_index(self.index, self.index_file)
            with open(self.chunk_file, 'w') as f:
                json.dump(self.chunks, f, indent=2)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.document_metadata, f)
            logger.info(f"Saved {len(self.chunks)} chunks and index to disk")
        except Exception as e:
            logger.error(f"Error saving to disk: {str(e)}")

    def _split_text(self, text: str, min_length: int = 20) -> List[str]:
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) >= min_length]
        final_chunks = []
        for para in paragraphs:
            if len(para) > 500:
                sentences = para.split('. ')
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) > 500:
                        if len(current_chunk) >= min_length:
                            final_chunks.append(current_chunk)
                        current_chunk = sent
                    else:
                        current_chunk += ". " + sent if current_chunk else sent
                if current_chunk and len(current_chunk) >= min_length:
                    final_chunks.append(current_chunk)
            else:
                final_chunks.append(para)
        return final_chunks

    def add_document(self, text: str, doc_id: str, doc_name: str, page_count: int = 1):
        try:
            paragraphs = self._split_text(text)
            print(" DEBUG: add_document() called")
            print(f"➡ OCR Text Sample: {text[:500]}")
            print(f"➡ Paragraph Count: {len(paragraphs)}")
            print(f"➡ Sample Paragraphs: {paragraphs[:2]}")

            if not paragraphs:
                logger.warning(f"No valid paragraphs found in document {doc_id}")
                return False

            embeddings = self.model.encode(paragraphs, show_progress_bar=False)
            self.index.add(np.array(embeddings))

            for i, para in enumerate(paragraphs):
                self.chunks.append({
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "text": para,
                    "page": (i % page_count) + 1,
                    "chunk_id": len(self.chunks),
                    "embedding_id": self.index.ntotal - len(paragraphs) + i
                })

            self.document_metadata[doc_id] = {
                "name": doc_name,
                "upload_time": datetime.now().isoformat(),
                "page_count": page_count,
                "chunk_count": len(paragraphs)
            }

            self._save_to_disk()
            return True

        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {str(e)}")
            return False

    def search(self, query: str, top_k: int = 5, doc_filter: Optional[List[str]] = None) -> List[Dict]:
        try:
            if self.index.ntotal == 0:
                logger.warning("Attempted search on empty index")
                return []

            query_embedding = self.model.encode([query], show_progress_bar=False)
            distances, indices = self.index.search(np.array(query_embedding), top_k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= len(self.chunks):
                    continue
                chunk = self.chunks[idx]
                if doc_filter and chunk['doc_id'] not in doc_filter:
                    continue
                results.append({
                    **chunk,
                    "score": float(1 - distances[0][i]),
                    "distance": float(distances[0][i])
                })
            return results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def get_document_chunks(self, doc_id: str) -> List[Dict]:
        return [chunk for chunk in self.chunks if chunk['doc_id'] == doc_id]

    def delete_document(self, doc_id: str) -> bool:
        try:
            if doc_id not in self.document_metadata:
                return False
            delete_indices = [i for i, chunk in enumerate(self.chunks) if chunk['doc_id'] == doc_id]
            if not delete_indices:
                return False
            all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            keep_mask = np.ones(self.index.ntotal, dtype=bool)
            keep_mask[[chunk['embedding_id'] for chunk in self.chunks if chunk['doc_id'] == doc_id]] = False
            new_index = faiss.IndexFlatL2(self.embedding_dim)
            new_index.add(all_vectors[keep_mask])
            self.index = new_index
            self.chunks = [chunk for chunk in self.chunks if chunk['doc_id'] != doc_id]
            del self.document_metadata[doc_id]
            for i, chunk in enumerate(self.chunks):
                chunk['embedding_id'] = i
            self._save_to_disk()
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            return False

#  This avoids circular import errors
vector_store = VectorStore()
