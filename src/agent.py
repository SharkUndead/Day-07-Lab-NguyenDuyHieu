from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # Bước 1: Truy xuất top-k chunk liên quan nhất từ store
        search_results = self.store.search(question, top_k=top_k)
        
        # Rút trích phần text (content) từ các kết quả tìm kiếm được
        context_chunks = [result["content"] for result in search_results]
        
        # Gộp các chunk lại thành một khối văn bản ngữ cảnh, cách nhau bởi 2 dấu xuống dòng
        context_text = "\n\n---\n\n".join(context_chunks)
        
        # Bước 2: Xây dựng prompt chứa ngữ cảnh và câu hỏi
        prompt = (
            "You are a helpful and accurate assistant. "
            "Please answer the user's question based strictly on the following context. "
            "If the context does not contain the answer, say 'I cannot find the answer in the provided context.'\n\n"
            f"Context Information:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        
        # Bước 3: Gọi hàm LLM (llm_fn) để sinh ra câu trả lời cuối cùng
        return self.llm_fn(prompt)