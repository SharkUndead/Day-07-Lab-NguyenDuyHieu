from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore
from src.chunking import FixedSizeChunker

# 1. Xử lý đường dẫn tuyệt đối để không bao giờ bị miss file
BASE_DIR = Path(__file__).parent.absolute()

SAMPLE_FILES = [
    str(BASE_DIR / "shopee_docs/dieu_khoan_dich_vu.txt"),
    str(BASE_DIR / "shopee_docs/chinh_sach_tra_hang_hoan_tien.txt"),
    str(BASE_DIR / "shopee_docs/quy_dinh_ve_dang_ban_san_pham_tren_shoppe.txt"), 
    str(BASE_DIR / "shopee_docs/chinh_sach_bao_mat.txt"),
    str(BASE_DIR / "shopee_docs/chinh_sach_cam_han_che_san_pham.txt"),
    str(BASE_DIR / "shopee_docs/chinh_sach_chong_gian_lan.txt")
]

def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths with utf-8 encoding."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)
        if path.suffix.lower() not in allowed_extensions:
            continue

        if not path.exists():
            print(f"❌ KHÔNG TÌM THẤY FILE: {path}")
            continue

        try:
            content = path.read_text(encoding="utf-8")
            documents.append(
                Document(
                    id=path.stem,
                    content=content,
                    metadata={"source": str(path), "extension": path.suffix.lower()},
                )
            )
        except Exception as e:
            print(f"❌ Lỗi khi đọc file {path.name}: {e}")

    return documents

def demo_llm(prompt: str) -> str:
    """Mock LLM to preview the context sent to the model."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."

def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    # Nạp cấu hình từ .env
    load_dotenv(override=True)
    
    files = sample_files or SAMPLE_FILES
    query = question or "Những sản phẩm nào bị cấm bán trên Shopee?"

    print("\n=== Manual File Test ===")
    docs = load_documents_from_files(files)
    if not docs:
        print("\n❌ Lỗi: Không có tài liệu nào được nạp thành công.")
        return 1

    print(f"✅ Đã load thành công {len(docs)} tài liệu Shopee.")

    # 2. Khởi tạo Embedder (OpenAI / Local / Mock)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "openai":
        try:
            model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
            embedder = OpenAIEmbedder(model_name=model_name)
            embedder("test connection") 
        except Exception as e:
            print(f"❌ LỖI OPENAI: {e}")
            print("👉 Tự động chuyển về Mock Embedder...")
            embedder = _mock_embed
    elif provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception as e:
            print(f"⚠️ Lỗi Local Embedder: {e}")
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"🚀 Backend thực tế đang dùng: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    # 3. CHUNKING: Chia nhỏ tài liệu trước khi tạo Embedding (Để tránh lỗi 8192 tokens)
    my_chunker = FixedSizeChunker(chunk_size=1000, overlap=150)
    
    # Tìm hàm chia nhỏ tài liệu trong class FixedSizeChunker
    possible_methods = ['chunk', 'split', 'split_text', 'chunk_document', 'split_document']
    chunk_fn = next((getattr(my_chunker, m) for m in possible_methods if hasattr(my_chunker, m)), None)
    
    if not chunk_fn:
        print("❌ Lỗi: Chunker không có hàm xử lý phù hợp.")
        return 1

    print(f"🔄 Đang thực hiện Chunking bằng hàm '{chunk_fn.__name__}'...")
    all_chunks = []
    
    for doc in docs:
        try:
            # THỬ THÁCH: Xử lý lỗi len() bằng cách thử gửi object, nếu lỗi thì gửi text
            try:
                raw_output = chunk_fn(doc)
            except (TypeError, AttributeError):
                raw_output = chunk_fn(doc.content)
            
            # Đóng gói lại kết quả thành list Document để Store hiểu được
            for i, item in enumerate(raw_output):
                if isinstance(item, str):
                    all_chunks.append(Document(
                        id=f"{doc.id}_chunk_{i}",
                        content=item,
                        metadata=doc.metadata.copy()
                    ))
                else:
                    all_chunks.append(item)
        except Exception as e:
            print(f"❌ Lỗi khi chia nhỏ file {doc.id}: {e}")

    # 4. Lưu trữ và Tìm kiếm
    store = EmbeddingStore(collection_name="shopee_manual_store", embedding_fn=embedder)
    print(f"💾 Đang tạo vector cho {len(all_chunks)} đoạn văn bản...")
    store.add_documents(all_chunks)

    print(f"\n=== EmbeddingStore Search Test ===")
    print(f"Câu hỏi: {query}")
    search_results = store.search(query, top_k=3)
    
    for index, result in enumerate(search_results, start=1):
        score_val = result.get('score', 0)
        source_name = Path(result['metadata'].get('source', 'Unknown')).name
        print(f"{index}. [Score={score_val:.3f}] Nguồn: {source_name}")
        print(f"   Nội dung: {result['content'][:150].strip().replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(agent.answer(query, top_k=3))
    
    return 0

def main() -> int:
    # Cho phép nhận câu hỏi từ Terminal
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(0)