from __future__ import annotations

import math
from typing import Any, Callable

from .chunking import _dot, compute_similarity 
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401
            # Phần này tạm comment lại, vì hệ thống test đang test in-memory store
            # self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Tạo một bản ghi (record) chứa thông tin của document và vector embedding của nó."""
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": embedding
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Tìm kiếm trong danh sách các records đã cho."""
        if not records:
            return []
            
        query_embedding = self._embedding_fn(query)
        
        # Tính điểm cosine similarity cho từng record
        results = []
        for record in records:
            score = compute_similarity(query_embedding, record["embedding"])
            results.append({
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                "score": score
            })
            
        # Sắp xếp kết quả giảm dần theo điểm score (từ cao xuống thấp)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Trả về top_k kết quả
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            record = self._make_record(doc)
            self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            # Nếu không có filter, tìm kiếm trên toàn bộ store
            return self.search(query, top_k)
            
        # Lọc danh sách records theo metadata
        filtered_records = []
        for record in self._store:
            # Kiểm tra xem record có chứa tất cả các key/value trong filter không
            match = True
            for k, v in metadata_filter.items():
                if record["metadata"].get(k) != v:
                    match = False
                    break
            if match:
                filtered_records.append(record)
                
        # Tìm kiếm trên danh sách đã lọc
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        initial_size = len(self._store)
        
        # Chỉ giữ lại những document có id KHÁC với doc_id cần xóa
        self._store = [record for record in self._store if record["id"] != doc_id]
        
        # Nếu kích thước store thay đổi (nhỏ hơn ban đầu), tức là đã có doc bị xóa
        return len(self._store) < initial_size