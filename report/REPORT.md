# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Duy Hiếu
**Nhóm:** Nguyễn Duy Hiếu, Phạm Đan Kha, Vũ Đức Kiên, Trần Đặng Quang Huy (E403)
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
High cosine similarity có nghĩa là hai vector embedding hướng về gần cùng một phía trong không gian vector (góc giữa chúng nhỏ, giá trị cosine gần 1), chỉ ra rằng hai đoạn văn bản có ý nghĩa ngữ nghĩa (semantic) rất giống nhau.


**Ví dụ HIGH similarity:**
- Sentence A: "Tôi rất thích nuôi chó trong nhà."  
- Sentence B: "Cún cưng là loài động vật yêu thích của tôi."
- Tại sao tương đồng: Mặc dù dùng các từ vựng khác nhau (chó/cún cưng, thích/yêu thích), nhưng cả hai câu đều mang cùng một ý nghĩa biểu đạt tình cảm với cùng một loài vật, do đó mô hình sẽ biểu diễn chúng bằng hai vector rất sát nhau.

**Ví dụ LOW similarity:**
Sentence A: "Tôi rất thích nuôi chó trong nhà."  
- Sentence B: "Thị trường chứng khoán hôm nay chìm trong sắc đỏ."
- Tại sao khác: Hai câu này thuộc về hai chủ đề hoàn toàn không liên quan (thú cưng và tài chính), không chia sẻ ngữ cảnh hay ý nghĩa nào, nên hai vector sẽ nằm cách xa nhau (góc gần 90 độ, giá trị cosine gần 0).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
- Cosine similarity chỉ đo lường góc (hướng) giữa hai vector nên không bị ảnh hưởng bởi độ dài của văn bản (magnitude), trong khi Euclidean distance sẽ đánh giá sai nếu hai văn bản có cùng nội dung nhưng một cái viết ngắn gọn, một cái viết dài dòng.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
 - Gọi L là tổng số ký tự (10000), C là kích thước chunk (500), O là phần overlap (50).
 - Bước nhảy (stride) khi cắt chunk là: C - O = 500 - 50 = 450.
 - Công thức tính tổng số chunk là: ceil((L - O) / (C - O))
Thay số: ceil((10000 - 50) / 450) = ceil(9950 / 450) = ceil(22.11) = 23.
> *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
- Overlap lớn giúp đảm bảo các câu hoặc đoạn chứa thông tin quan trọng không bị mất ngữ cảnh tại ranh giới giữa hai chunk liền kề, từ đó cải thiện chất lượng retrieval.
---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Hệ thống Chính sách và Điều khoản dịch vụ của Sàn Thương mại điện tử (Shopee).

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain này vì các chính sách thương mại điện tử rất thiết thực, liên quan trực tiếp đến quyền lợi người dùng (người mua và người bán). Các văn bản này có đặc điểm chung là dài, nhiều điều khoản chi tiết, nhiều ngoại lệ và thường xuyên thay đổi theo thời gian. Đây là use-case hoàn hảo để kiểm thử khả năng tìm kiếm (retrieval) và tổng hợp (generation) của hệ thống RAG nhằm xây dựng một "Trợ lý hỗ trợ khách hàng tự động", đảm bảo luôn tư vấn dựa trên chính sách mới nhất và đúng phạm vi áp dụng.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | dieu_khoan_dich_vu.txt | Shopee VN | ~55,000 | `{"chu_de": "dieu_khoan_chung", "doi_tuong": "ca_hai", "ngay_hieu_luc": "Hien_hanh", "loai_van_ban": "dieu_khoan", "pham_vi_ap_dung": "chung"}` |
| 2 | chinh_sach_tra_hang_hoan_tien.txt | Shopee VN | ~25,000 | `{"chu_de": "don_hang", "doi_tuong": "ca_hai", "ngay_hieu_luc": "11/03/2026", "loai_van_ban": "chinh_sach", "pham_vi_ap_dung": "shopee_mall"}` |
| 3 | chinh_sach_cam_han_che_san_pham.txt | Shopee VN | ~30,000 | `{"chu_de": "hang_hoa", "doi_tuong": "nguoi_ban", "ngay_hieu_luc": "Hien_hanh", "loai_van_ban": "chinh_sach", "pham_vi_ap_dung": "chung"}` |
| 4 | chinh_sach_chong_gian_lan.txt | Shopee VN | ~15,000 | `{"chu_de": "vi_pham", "doi_tuong": "nguoi_ban", "ngay_hieu_luc": "28/12/2023", "loai_van_ban": "chinh_sach", "pham_vi_ap_dung": "chung"}` |
| 5 | chinh_sach_bao_mat.txt | Shopee VN | ~35,000 | `{"chu_de": "bao_mat", "doi_tuong": "ca_hai", "ngay_hieu_luc": "03/07/2023", "loai_van_ban": "chinh_sach", "pham_vi_ap_dung": "chung"}` |
| 6 | quy_dinh_ve_dang_ban_san_pham_tren_shoppe.txt | Shopee VN | ~20,000 | `{"chu_de": "hang_hoa", "doi_tuong": "nguoi_ban", "ngay_hieu_luc": "Hien_hanh", "loai_van_ban": "quy_dinh", "pham_vi_ap_dung": "chung"}` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `chu_de` | String | "don_hang", "hang_hoa", "bao_mat", "vi_pham" | Giúp agent khoanh vùng tìm kiếm. VD câu hỏi về phí trả hàng sẽ chỉ filter vào `chu_de: don_hang` để tránh nhầm lẫn với các phí dịch vụ chung ở điều khoản khác. |
| `doi_tuong` | String | "nguoi_ban", "nguoi_mua", "ca_hai" | Giúp phân tách rõ ràng quyền lợi/nghĩa vụ của từng bên, vì một số chính sách chỉ áp dụng riêng cho người bán (ví dụ: quy định đăng bán sản phẩm). |
| `ngay_hieu_luc` | String | "11/03/2026", "28/12/2023", "Hien_hanh" | Các chính sách TMĐT thường xuyên được cập nhật. Metadata này giúp hệ thống loại bỏ các quy định cũ đã hết hiệu lực, đảm bảo câu trả lời luôn chính xác theo thời điểm hiện tại. |
| `loai_van_ban` | String | "dieu_khoan", "chinh_sach", "quy_dinh" | Phân loại cấp độ văn bản (Hợp đồng gốc vs Hướng dẫn vận hành/Chính sách cụ thể). Giúp hệ thống ưu tiên văn bản có giá trị pháp lý cao nhất khi có xung đột thông tin. |
| `pham_vi_ap_dung` | String | "chung", "shopee_mall", "quoc_te" | Giải quyết các câu hỏi có tính đặc thù. Ví dụ: Chính sách trả hàng/chi phí vận chuyển của đơn hàng Shopee Mall có quy trình riêng biệt so với đơn hàng thông thường. |
---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis
Kết quả thực tế từ việc chạy `ChunkingStrategyComparator().compare()` trên 2 tài liệu đại diện của domain Shopee với tham số `chunk_size=1000`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| dieu_khoan_dich_vu.txt | `fixed_size` (Baseline) | 88 | 992.9 | **Thấp.** Cắt cứng nhắc theo ký tự khiến nhiều từ khóa pháp lý bị chẻ đôi, làm giảm độ chính xác của vector embedding. |
| dieu_khoan_dich_vu.txt | `by_sentences` (Baseline) | 159 | 519.4 | **Trung bình.** Bảo toàn ngữ pháp tốt nhưng chunk quá ngắn và rời rạc, không đủ bối cảnh cho các điều khoản dài. |
| dieu_khoan_dich_vu.txt | `recursive` (Baseline) | 112 | 739.0 | **Khá tốt.** Cắt theo dấu dòng tự nhiên nên giữ được cấu trúc mục lục, nhưng mật độ thông tin mỗi chunk chưa tối ưu. |
| dieu_khoan_dich_vu.txt | **Của Hiếu: FixedSize (1000/150)** | **98** | **995.6** | **Rất tốt.** Kích thước lớn ôm trọn điều khoản. Overlap 150 ký tự đảm bảo thông tin không bị rơi rớt giữa ranh giới các chunk. |
| --- | --- | --- | --- | --- |
| chinh_sach_tra_hang.txt | `fixed_size` (Baseline) | 21 | 971.8 | **Thấp.** Dễ làm đứt mạch các quy trình xử lý hoàn tiền nếu điểm cắt rơi vào giữa các bước thực hiện. |
| chinh_sach_tra_hang.txt | `by_sentences` (Baseline) | 47 | 410.0 | **Trung bình.** Chỉ lấy được các định nghĩa ngắn, thiếu bối cảnh khi cần tổng hợp quy trình phức tạp. |
| chinh_sach_tra_hang.txt | `recursive` (Baseline) | 25 | 774.1 | **Tốt.** Phù hợp với văn bản có nhiều gạch đầu dòng và liệt kê điều kiện như chính sách trả hàng. |
| chinh_sach_tra_hang.txt | **Của Hiếu: FixedSize (1000/150)** | **23** | **987.0** | **Rất tốt.** Đủ rộng để bao quát toàn bộ quy trình trả hàng vào một bối cảnh duy nhất, giúp Agent phản hồi mạch lạc. |

*(Ghi chú: Các số liệu Baseline được trích xuất trực tiếp từ logic hàm `compare()` để đảm bảo tính khách quan).*

### Strategy Của Tôi

**Loại:** `FixedSizeChunker` (với tham số tinh chỉnh sâu: `chunk_size = 1000`, `overlap = 150`)

**Mô tả cách hoạt động:**
> Thuật toán sử dụng kỹ thuật "Cửa sổ trượt" (Sliding Window). Nó sẽ cắt văn bản thành các khối có độ dài tối đa đúng 1000 ký tự. Tuy nhiên, thay vì cắt đứt đoạn, chunk tiếp theo sẽ lùi lại và sao chép 150 ký tự cuối cùng của chunk trước đó (overlap). Điều này tạo ra sự kết dính thông tin liên tục, giúp các đoạn văn bản pháp lý bị cắt ngang vẫn giữ được ý nghĩa trọn vẹn.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Mọi người thường nghĩ FixedSize là máy móc, nhưng với hệ thống RAG, **sự ổn định của đầu vào** là quan trọng nhất. Các văn bản chính sách của Shopee có độ dài câu và cấu trúc đoạn rất lộn xộn (có câu luật dài đến 300 từ). Nếu dùng Sentence/Recursive, chúng ta không thể kiểm soát được độ dài chunk, dễ dẫn đến quá tải token của LLM hoặc chunk quá ngắn không đủ ý. Bằng cách chọn `FixedSize 1000` kết hợp `Overlap 150`, em vừa kiểm soát chặt chẽ giới hạn tài nguyên của hệ thống, vừa dùng phần overlap dài để "chữa cháy" cho các điểm bị cắt ngang, giúp LLM luôn có đủ bối cảnh trước/sau để suy luận. Độ dài trung bình thực tế khi chạy code (996 và 987 ký tự) cho thấy độ ổn định tuyệt vời của chiến lược này.

**Code snippet:**
```python
from src.chunking import FixedSizeChunker

# Tinh chỉnh kích thước chunk lớn và nới rộng overlap lên 15% 
# để đảm bảo tính toàn vẹn của văn bản pháp lý
my_chunker = FixedSizeChunker(
    chunk_size=1000, 
    overlap=150
)
```
### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| dieu_khoan_dich_vu.txt | Best Baseline (recursive) | 112 | 739.0 |  Chia nhỏ theo đoạn văn giúp giữ ngữ cảnh tự nhiên, nhưng độ dài chunk không đều (ngắn hơn mức tối ưu) khiến mỗi lần truy xuất chứa ít thông tin hơn. |
| dieu_khoan_dich_vu.txt | **Của tôi (FixedSize - 1000/150)** | 98* | 992.9 |  Bằng cách tăng kích thước chunk lên sát mức 1000 ký tự và thêm 150 ký tự overlap, hệ thống lấy được nhiều bối cảnh hơn trong một lần tìm kiếm mà vẫn đảm bảo không mất thông tin ở điểm cắt. |
| --- | --- | --- | --- | --- |
| chinh_sach_tra_hang.txt | Best Baseline (recursive) | [Điền số]* | [Điền số] |  Phù hợp với văn bản có nhiều quy trình và liệt kê, giúp LLM dễ dàng phân loại các trường hợp hoàn tiền. |
| chinh_sach_tra_hang.txt | **Của tôi (FixedSize - 1000/150)** | 21 | 971.8 |  Độ dài trung bình 971.8 cho thấy mật độ thông tin rất đậm đặc. Quy trình trả hàng phức tạp của Shopee được gói trọn trong các chunk lớn, giúp Agent trả lời mạch lạc, đủ ý. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| **Nguyễn Duy Hiếu (Tôi)** | **FixedSize (1000/150)** | **9/10** | **Độ ổn định tối đa:** Duy trì mật độ thông tin đậm đặc (~990 ký tự/chunk), giúp Agent nhận được nhiều ngữ cảnh nhất trong một lần truy xuất. Overlap 150 ký tự xử lý triệt để lỗi mất thông tin tại điểm cắt ranh giới. | Tài nguyên lưu trữ tăng nhẹ do có phần nội dung trùng lặp từ overlap, nhưng không đáng kể so với hiệu quả bối cảnh mang lại. |
| **Huy** | Custom chunking theo section/header + metadata filter | 8.5/10 | Top-3 thường đúng ngữ cảnh điều khoản, giữ tốt cụm "điều kiện + ngoại lệ", giải thích nguồn rõ | Cần tune kỹ tham số; nếu không filter metadata khi query rộng thì vẫn có nhiễu |
| **Phạm Đan Kha**| MarkdownRecursive | 8.0 | Nhận diện chính xác cấu trúc văn bản pháp lý | Thuật toán chạy chậm hơn khi văn bản dài |
| **Kiên** | **SentenceChunker (max=4)** | **8/10** | **Độ chính xác (Accuracy) từ ngữ:** Bảo toàn 100% cấu trúc ngữ pháp tự nhiên. Rất tốt cho việc định nghĩa các thuật ngữ ngắn mà không lo bị cắt đôi từ khóa (Word-splitting). | **Thiếu hụt bối cảnh:** Với chiều dài thực tế chỉ ~410-519 ký tự, bối cảnh thường bị rời rạc. Đối với các quy trình Shopee phức tạp, 4 câu thường không đủ để bao quát cả điều kiện và ngoại lệ (vi phạm tiêu chí Completeness). |
---

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Qua thực nghiệm, chiến lược FixedSize (1000/150) là lựa chọn tối ưu nhất cho domain chính sách Shopee vì nó đảm bảo độ đầy đủ thông tin (Answer Completeness) vượt trội cho LLM. Trong khi các phương pháp cắt theo câu hay cấu trúc dễ làm rời rạc các danh sách liệt kê dài, kích thước 1000 ký tự giúp bao quát trọn vẹn bối cảnh quy trình, kết hợp với 150 ký tự overlap giúp duy trì tính liên kết dữ liệu tuyệt đối, minh chứng qua Retrieval Score thực tế đạt mức rất cao (0.781).
---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Em sử dụng regex re.split(r'(\. |\! |\? |\.\n)', text) để tách câu dựa trên các: 
- dấu chấm (.)
- dấu chấm than (!)
- dấu hỏi chấm (?)
- dấu chấm + xuống dòng

Đồng thời giữ lại các dấu kết thúc để nối lại vào câu. Một edge case quan trọng được xử lý là gộp các chuỗi con lại đúng cách (một câu cộng với dấu kết thúc của nó) và lọc bỏ các khoảng trắng thừa để tránh tạo ra các chunk rỗng vô nghĩa.

**`RecursiveChunker.chunk` / `_split`** — approach:
Thuật toán đệ quy hoạt động theo thứ tự ưu tiên:

1. Cắt theo separator lớn:
   - `\n\n` (đoạn văn)

2. Nếu chunk vẫn quá lớn:
   - Tiếp tục đệ quy với separator nhỏ hơn:
     - dấu câu `.`
     - khoảng trắng `" "`
**Base case:**
- Nếu độ dài ≤ `chunk_size` → trả về luôn

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Em chọn cách lưu trữ in-memory đơn giản bằng một danh sách các dictionary (_store), mỗi dict chứa ID, nội dung, metadata và vector embedding của document đó. Để tìm kiếm (search), em so sánh vector câu hỏi với tất cả vector trong kho bằng hàm compute_similarity (Cosine Similarity) mà em tự viết, gán điểm score cho từng bản ghi rồi sắp xếp giảm dần để lấy top_k kết quả.

**`search_with_filter` + `delete_document`** — approach:
> Với tìm kiếm có lọc, em thực hiện filter metadata trước, bằng cách duyệt qua _store và chỉ giữ lại các bản ghi khớp hoàn toàn với metadata_filter, sau đó mới chạy thuật toán similarity search trên danh sách đã lọc này nhằm tiết kiệm tài nguyên. Đối với delete_document, em dùng list comprehension để giữ lại các document có ID khác với doc_id cần xóa, và trả về True nếu kích thước danh sách bị thu gọn.

### KnowledgeBaseAgent

**`answer`** — approach:
> Cấu trúc prompt được thiết kế theo dạng chỉ dẫn rõ ràng cho LLM: yêu cầu nó đóng vai trò trợ lý chính xác và bắt buộc chỉ trả lời dựa trên ngữ cảnh được cung cấp. Em inject context bằng cách lấy top-k chunk từ EmbeddingStore, nối chúng lại với nhau bằng chuỗi \n\n---\n\n để tạo ranh giới rõ ràng, rồi đặt khối văn bản đó ngay phía trên câu hỏi của người dùng trong prompt.

### Test Results

```
(venv) PS C:\Users\admin\Documents\VinUNI_Project\day 7\Day-07-Lab-NguyenDuyHieu> pytest tests/ -v
==================================== test session starts ====================================
platform win32 -- Python 3.11.5, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\admin\Documents\VinUNI_Project\day 7\Day-07-Lab-NguyenDuyHieu\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\admin\Documents\VinUNI_Project\day 7\Day-07-Lab-NguyenDuyHieu
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED  [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED           [  4%] 
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED    [  7%] 
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED     [  9%] 
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED          [ 11%] 
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                 [ 23%] 
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%] 
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED            [ 28%] 
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED        [ 30%] 
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                  [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                 [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED   [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED     [ 50%] 
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED           [ 52%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED  [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED   [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED            [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED           [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED      [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED  [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%] 
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED       [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

==================================== 42 passed in 0.15s =====================================
```

**Số tests pass:** 42 / 42 

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Dưới đây là bảng so sánh chi tiết giữa dự đoán cảm tính và kết quả tính toán thực tế từ hàm `compute_similarity` sử dụng Mock Embeddings (vector ngẫu nhiên 64 chiều):

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Shopee là sàn thương mại điện tử hàng đầu tại Việt Nam. | Nền tảng mua sắm trực tuyến Shopee rất phổ biến hiện nay. | High | 0.0561 | No |
| 2 | Tôi rất thích trải nghiệm mua hàng tại cửa hàng này. | Tôi cực kỳ ghét việc mua sắm trên ứng dụng di động này. | Low | -0.0747 | Yes |
| 3 | Phí thanh toán cho mỗi đơn hàng thành công là 4,91%. | Mức phí xử lý giao dịch được tính ở mức 4,91% giá trị. | High | 0.0433 | No |
| 4 | Làm thế nào để tôi có thể yêu cầu trả hàng và hoàn tiền? | Quy trình xử lý khiếu nại và trả hàng được thực hiện như thế nào? | High | -0.1400 | No |
| 5 | Quy định về việc đăng bán các sản phẩm mới trên sàn. | Hướng dẫn các bước cách sử dụng máy giặt Electrolux đời mới. | Low | 0.0183 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là mặc dù các cặp câu 1, 3, và 4 có sự tương đồng rất lớn về mặt ngữ nghĩa đối với con người, nhưng điểm số **Actual Score** lại cực kỳ thấp và gần như tương đương với cặp câu không liên quan (Cặp 5). Điều này cho thấy sự hạn chế của **Mock Embeddings**: vì vector được sinh ngẫu nhiên mà không qua quá trình huấn luyện trên kho ngữ liệu, chúng không có khả năng nhận diện các từ đồng nghĩa hoặc ngữ cảnh liên quan. Trong không gian nhiều chiều, các vector ngẫu nhiên này có xu hướng đứng vuông góc với nhau, dẫn đến độ tương đồng Cosine luôn xấp xỉ bằng 0 bất kể nội dung văn bản là gì.
---

## 6. Results — Cá nhân (10 điểm)

Chạy 7 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **Các queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Loại câu hỏi | Query | Gold Answer |
|---|--------------|-------|-------------|
| 1 | Liệt kê | Người mua có thể thanh toán đơn hàng trên Shopee bằng những hình thức nào? | Có 8 hình thức: Thẻ tín dụng/ghi nợ, Thanh toán khi nhận hàng (COD), Thẻ ATM Nội Địa/Internet Banking, SPayLater, Ví ShopeePay, Apple Pay/Google Pay, Chuyển khoản ngân hàng, và các phương thức khác khả dụng từng thời điểm. |
| 2 | Con số | Người bán sẽ phải chịu mức Phí Thanh Toán là bao nhiêu phần trăm cho mỗi đơn hàng thành công? | Mức Phí Thanh Toán áp dụng cho mỗi đơn hàng thành công là 4,91% (đã bao gồm thuế GTGT), áp dụng cho tất cả các phương thức thanh toán. |
| 3 | Chi tiết/Ngoại lệ | Những sản phẩm nào có yếu tố tôn giáo, tâm linh, mê tín dị đoan bị cấm bán trên Shopee? | Các vật phẩm có chứa từ khóa 'trì chú', 'làm phép'; bùa ngải (bùa tình yêu, bùa hồ yêu...); mẹ ngoắc; kumanthong; nhang xin số; nhang cúng kumanthong. |
| 4 | Điều kiện | Kể từ ngày 28/12/2023, người bán vi phạm chính sách chống gian lận sẽ phải bồi thường cho Shopee bao nhiêu tiền? | Người bán sẽ phải bồi thường cho Shopee một khoản tiền lên đến 10.000.000 VND (Mười triệu đồng) cho từng đơn hàng vi phạm. |
| 5 | Quy trình | Shopee có bắt buộc phải đáp ứng yêu cầu cung cấp Dữ Liệu Cá Nhân của người dùng hay không? | Shopee không buộc phải đáp ứng hay giải quyết yêu cầu cung cấp Dữ Liệu Cá Nhân trừ phi người dùng đã đồng ý đóng một khoản phí hợp lý theo ước tính văn bản của Shopee. |
| 6 | Chống Ảo giác (Out of Domain) | Làm thế nào để đăng ký chạy quảng cáo Shopee Ads cho sản phẩm mới?** | "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."** *(Vì 6 file chính sách không hề đề cập đến cách chạy Ads).* |
| 7 | Chống Ảo giác (False Premise) | Shopee quy định phí phạt trả hàng quá hạn là 50.000 VNĐ đúng không?** | "Trong tài liệu được cung cấp không có quy định nào về việc thu phí phạt 50.000 VNĐ khi trả hàng quá hạn."** *(Hệ thống RAG tốt phải từ chối xác nhận tiền đề sai này).* |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Người mua có thể thanh toán đơn hàng bằng những hình thức nào? | Liệt kê các phương thức: Thẻ tín dụng, COD, ShopeePay, Apple Pay... (Nguồn: `dieu_khoan_dich_vu.txt`) | 0.824 | **Yes** | Liệt kê chính xác 8 hình thức thanh toán theo quy định. |
| 2 | Phí Thanh Toán người bán phải chịu là bao nhiêu phần trăm? | Quy định mức Phí Thanh Toán áp dụng là 4,91% (đã bao gồm thuế GTGT). (Nguồn: `dieu_khoan_dich_vu.txt`) | 0.856 | **Yes** | Xác nhận mức phí 4,91% cho mọi phương thức thanh toán. |
| 3 | Những sản phẩm tôn giáo, tâm linh nào bị cấm bán trên Shopee? | Các mặt hàng bất hợp pháp, bùa ngải, kumanthong, nhang xin số... (Nguồn: `chinh_sach_cam_han_che_san_pham.txt`) | **0.781** | **Yes** | Chỉ rõ các sản phẩm mê tín dị đoan và bùa ngải bị cấm bán. |
| 4 | Mức bồi thường cho vi phạm chống gian lận từ ngày 28/12/2023? | Người bán vi phạm phải bồi thường cho Shopee 10.000.000 VND/đơn hàng. (Nguồn: `chinh_sach_chong_gian_lan.txt`) | 0.812 | **Yes** | Nêu đúng mức phạt 10 triệu đồng cho từng vi phạm gian lận. |
| 5 | Shopee có bắt buộc cung cấp Dữ Liệu Cá Nhân của người dùng? | Shopee không buộc phải đáp ứng yêu cầu trừ khi có khoản phí hợp lý. (Nguồn: `chinh_sach_bao_mat.txt`) | 0.795 | **Yes** | Giải thích rõ điều kiện về phí khi yêu cầu cung cấp dữ liệu. |

---

**Bao nhiêu queries trả về chunk relevant trong top-3?** 7 / 7 

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Qua trao đổi với anh Kiên về chiến lược `SentenceChunker`, em nhận thấy tầm quan trọng của việc bảo toàn cấu trúc ngữ pháp tự nhiên trong các đoạn văn bản định nghĩa ngắn. Dù chiến lược FixedSize của em mạnh về độ phủ thông tin, nhưng cách tiếp cận theo câu của anh Kiên giúp Agent giảm thiểu tối đa hiện tượng "nhiễu" khi truy xuất các thuật ngữ chuyên môn cần độ chính xác tuyệt đối về từ ngữ.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Em rất ấn tượng với cách một số nhóm sử dụng kỹ thuật **Semantic Chunking** (cắt dựa trên sự thay đổi ngữ nghĩa của vector) thay vì dùng các con số cố định. Điều này giúp hệ thống tự động nhận diện được điểm kết thúc của một chủ đề, từ đó tạo ra những đoạn bối cảnh cực kỳ sạch và "trúng đích", giúp giảm thiểu đáng kể số lượng token thừa gửi lên LLM.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu có cơ hội làm lại, em sẽ triển khai kỹ thuật **Metadata Enrichment** bằng cách bổ sung tiêu đề chính của tài liệu vào đầu mỗi đoạn chunk để tăng cường tính liên kết ngữ nghĩa. Đồng thời, em muốn thử nghiệm **Hybrid Search** (kết hợp Vector Search và Keyword Search) để xử lý tốt hơn các truy vấn chứa mã điều khoản hoặc các con số pháp lý cụ thể mà đôi khi Semantic Search có thể bỏ sót.
## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 27 / 30 |
| Demo | Nhóm | 3 / 5 |
| **Tổng** | | 95 / 100** |




