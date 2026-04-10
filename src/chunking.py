from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
            
        # Tách văn bản dựa trên các dấu hiệu kết thúc câu, giữ lại dấu câu
        splits = re.split(r'(\. |\! |\? |\.\n)', text)
        
        sentences = []
        current_sentence = ""
        
        for i, part in enumerate(splits):
            current_sentence += part
            # Regex trả về text ở index chẵn, delimiter ở index lẻ
            # Nếu là delimiter (hoặc index cuối cùng nhưng không có delimiter), ta chốt câu
            if i % 2 != 0 or i == len(splits) - 1:
                cleaned = current_sentence.strip()
                if cleaned:
                    sentences.append(cleaned)
                current_sentence = ""
                
        # Gom nhóm các câu thành chunk
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(chunk_group))
            
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case: Nếu độ dài đã phù hợp thì giữ nguyên
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Tìm separator phù hợp nhất (ưu tiên cao nhất và có tồn tại trong text)
        separator = ""
        next_separators = []
        for i, sep in enumerate(remaining_separators):
            if sep == "":
                separator = sep
                break
            if sep in current_text:
                separator = sep
                next_separators = remaining_separators[i + 1:]
                break

        # Nếu không có separator nào (hoặc dùng separator rỗng ""), cắt cứng theo số lượng ký tự
        if separator == "":
            return [
                current_text[i : i + self.chunk_size] 
                for i in range(0, len(current_text), self.chunk_size)
            ]

        splits = current_text.split(separator)
        final_chunks = []
        current_chunk = ""

        for s in splits:
            # Nếu bản thân 1 split đã quá dài, đệ quy trực tiếp nó với các separator mức thấp hơn
            if len(s) > self.chunk_size:
                if current_chunk:
                    final_chunks.append(current_chunk)
                    current_chunk = ""
                final_chunks.extend(self._split(s, next_separators))
                continue

            # Tính toán xem nếu nối thêm vào chunk hiện tại thì có vượt quá giới hạn không
            proposed_len = len(current_chunk) + len(separator) + len(s) if current_chunk else len(s)
            
            if proposed_len > self.chunk_size:
                final_chunks.append(current_chunk)
                current_chunk = s
            else:
                if current_chunk:
                    current_chunk += separator + s
                else:
                    current_chunk = s

        if current_chunk:
            final_chunks.append(current_chunk)

        # Lọc bỏ các chunk rỗng sinh ra do split thừa khoảng trắng
        return [c.strip() for c in final_chunks if c.strip()]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0
        
    dot_product = _dot(vec_a, vec_b)
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
        
    return dot_product / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed = FixedSizeChunker(chunk_size=chunk_size)
        sentence = SentenceChunker(max_sentences_per_chunk=3)
        recursive = RecursiveChunker(chunk_size=chunk_size)

        fixed_chunks = fixed.chunk(text)
        sentence_chunks = sentence.chunk(text)
        recursive_chunks = recursive.chunk(text)

        # Tránh lỗi chia cho 0 khi tính trung bình (phòng trường hợp list chunks rỗng)
        def _get_avg(chunks_list: list[str]) -> float:
            if not chunks_list:
                return 0.0
            return sum(len(c) for c in chunks_list) / len(chunks_list)

        return {
            "fixed_size": {
                "count": len(fixed_chunks),
                "avg_length": _get_avg(fixed_chunks),
                "chunks": fixed_chunks
            },
            "by_sentences": {
                "count": len(sentence_chunks),
                "avg_length": _get_avg(sentence_chunks),
                "chunks": sentence_chunks
            },
            "recursive": {
                "count": len(recursive_chunks),
                "avg_length": _get_avg(recursive_chunks),
                "chunks": recursive_chunks
            }
        }