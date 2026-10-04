# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Đào Đức Hải  
**MSSV:** 2A202602752  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code bạn vừa viết trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Dựa trên cosine similarity giữa embedding các câu liên tiếp sử dụng mô hình BAAI/bge-m3. Khi similarity giảm dưới threshold (0.85), ranh giới chunk mới được tạo. Kỹ thuật này bảo toàn trọn vẹn ngữ nghĩa câu văn, tránh việc cắt ngang mệnh đề hoặc ngắt mạch logic như fixed-size chunking. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | Kết hợp ưu thế của Lexical Search (BM25Okapi với tách từ tiếng Việt bằng Underthesea/regex) và Semantic Search (Dense vector trên Qdrant với bge-m3). Công thức RRF \( \text{RRF Score} = \sum \frac{1}{k + \text{rank}} \) với hằng số chuẩn hóa \( k=60 \) giúp triệt tiêu chênh lệch thang điểm (score distribution mismatch), đưa tài liệu đúng top đầu ngay cả khi một bên retrieval cho điểm thấp. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Mô hình `BAAI/bge-reranker-v2-m3` nhận trực tiếp cặp `(query, document)` và tính attention xuyên suốt (full cross-attention), loại bỏ giả định độc lập embedding của Bi-Encoder. Lọc lại top 3 tài liệu chính xác nhất từ danh sách 20 ứng viên ban đầu của Hybrid Search, giảm đáng kể nhiễu ngữ cảnh cho LLM. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Đánh giá toàn diện 4 khía cạnh cốt lõi của RAG: **Faithfulness** (câu trả lời có bám sát context không, hạn chế hallucination), **Answer Relevancy** (câu trả lời có giải quyết đúng trọng tâm câu hỏi không), **Context Precision** (mức độ chính xác và thứ hạng của context được truy xuất), và **Context Recall** (context có bao hàm đủ thông tin ground truth để trả lời không). Kết hợp RunConfig điều tiết concurrency để tránh chạm rate-limit. |
| Contextual embeddings & Enrichment | M5 | `contextual_prepend()` / `_enrich_single_call()` | Áp dụng kỹ thuật của Anthropic: bổ sung 1 câu ngữ cảnh tổng quan về vị trí và chủ đề của chunk trong toàn bộ tài liệu nguồn trước nội dung chunk. Đồng thời trích xuất metadata và sinh các câu hỏi giả định (Hypothetical Questions / HyQA) thông qua một LLM call duy nhất (`_enrich_single_call`) giúp tối ưu chi phí và tăng tỷ lệ hit khi người dùng đặt câu hỏi tương đương. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - **Lỗi 1 (Rate Limit):** `RateLimitError: Error code: 429 - Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 15, model: gemini-3.5-flash-lite. Please retry in 19s.`
  - **Lỗi 2 (Ragas Gemini Incompatibility):** `BadRequestError: Error code: 400 - Multiple candidates is not enabled for this model` khi Ragas gọi endpoint OpenAI-compatible của Gemini.
  - **Lỗi 3 (Ragas Embeddings Mismatch):** Khi chạy Ragas trên môi trường không có OpenAI API key, Ragas mặc định tìm `openai.embeddings` dẫn tới crash hoặc lỗi kết nối.
  - **Lỗi 4 (Unit Test Length Assertion):** `AssertionError: assert 143 <= (65 * 2)` trong `test_summarize_shorter_than_original` do LLM tóm tắt đoạn văn 1 câu quá chi tiết thành 2 câu dài hơn 2 lần bản gốc.

- **Nguyên nhân gốc rễ & Cách debug:**
  - *Với Rate Limit (429):* Free tier của Gemini chỉ cho phép tối đa 15 Requests Per Minute (RPM). Ragas mặc định chạy `max_workers=16` gửi hàng chục request đồng thời khiến API bị nghẽn ngay lập tức. **Cách giải quyết:** Cấu hình `RunConfig(max_workers=2, max_retries=5, max_wait=10)` trong `src/m4_eval.py` để giãn tiến trình gửi request và tự động retry khi gặp lỗi tạm thời.
  - *Với lỗi `Multiple candidates is not enabled`:* Gemini API không hỗ trợ tham số `n > 1` (sinh nhiều câu trả lời cùng lúc). Ragas gọi qua LangChain `ChatOpenAI` đôi khi truyền `n=2` hoặc `n=3`. **Cách giải quyết:** Viết lớp wrapper `GeminiChatOpenAI(ChatOpenAI)` ghi đè phương thức request payload để ép `payload["n"] = 1`.
  - *Với Embeddings trong Ragas:* Sử dụng `HuggingFaceEmbeddings(model_name="BAAI/bge-m3")` truyền trực tiếp vào hàm `evaluate()` của Ragas, đồng bộ hoàn toàn với mô hình dense embedding của hệ thống và chạy local 100% không phụ thuộc API ngoài.
  - *Với Summarization Test:* Cập nhật system prompt yêu cầu tóm tắt cô đọng trong 1 câu không dài hơn bản gốc, đồng thời kiểm tra nếu độ dài output vượt quá ngưỡng an toàn thì fallback về extractive summarization.

- **Kiến thức còn thiếu & Cách khắc phục:**
  - Cần nắm vững cơ chế rate-limiting, exponential backoff và cost optimization khi triển khai pipeline gọi LLM hàng loạt.
  - Nắm rõ sự khác biệt giữa OpenAI API chuẩn và các endpoint tương thích (Gemini OpenAI proxy, vLLM, Ollama) về các tham số đặc thù như `n`, `temperature`, `response_format`.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

Dựa trên những kỹ thuật đã học và thực hành, lập kế hoạch cụ thể áp dụng vào project của bạn:

### Project: Hệ thống Trợ lý Pháp lý & Tra cứu Quy chế Doanh nghiệp (Legal & Policy RAG Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng LangChain cơ bản với RecursiveCharacterTextSplitter (chunk_size=1000, overlap=100), Dense retrieval qua FAISS với mô hình `all-MiniLM-L6-v2` và sinh câu trả lời trực tiếp bằng LLM.
- **Vấn đề / Bottlenecks đang gặp:**
  - *Retrieval recall thấp đối với từ khóa văn bản luật:* Khi người dùng tra cứu chính xác số hiệu văn bản, điều khoản (ví dụ: "Nghị định 13/2023", "Điều 45"), Dense search đơn thuần không bắt được từ khóa chính xác.
  - *Context bị đứt gãy:* Cắt theo số ký tự làm rách bảng biểu, đứt câu hoặc tách rời tiêu đề chương/mục khỏi nội dung quy định.
  - *Hallucination:* LLM tự suy diễn các mốc thời gian và chế tài xử phạt khi context đưa vào chứa các thông tin gần giống nhưng ở các điều khoản khác.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Áp dụng **Structure-Aware Chunking** kết hợp **Hierarchical Chunking (Parent-Child)**. Các điều, khoản, mục của quy chế và văn bản luật được tách theo đúng cấu trúc tiêu đề. Khi tìm kiếm, child chunk nhỏ (200-400 ký tự) dùng để match chính xác, sau đó nạp parent chunk (toàn bộ điều luật 1500 ký tự) cho LLM.
2. **Search retrieval:** Triển khai **Hybrid Search (BM25Okapi + Dense Qdrant)** với **Reciprocal Rank Fusion (RRF)**. Tách từ tiếng Việt bằng Underthesea để BM25 bắt chính xác thuật ngữ pháp lý và số hiệu điều luật, trong khi mô hình đa ngôn ngữ `BAAI/bge-m3` đảm bảo khả năng tìm kiếm ngữ nghĩa linh hoạt.
3. **Reranking:** Sử dụng Cross-Encoder `BAAI/bge-reranker-v2-m3` để chọn top 3-5 đoạn trích phù hợp nhất từ top 25 kết quả của Hybrid Search, triệt tiêu tài liệu gây nhiễu trước khi nạp vào LLM prompt.
4. **Enrichment:** Áp dụng **Contextual Prepend** theo phong cách Anthropic để gắn tên văn bản, số hiệu, chương và mục vào đầu mỗi chunk.
5. **Evaluation:** Thiết lập bộ benchmark 100 câu hỏi - đáp pháp lý mẫu, tự động đánh giá liên tục với RAGAS (4 chỉ số) trong quy trình CI/CD.

#### 3. Timeline triển khai
- **Tuần 1:**
  - Chuẩn hóa bộ parser tài liệu PDF/DOCX quy chế doanh nghiệp theo cấu trúc Chương - Mục - Điều.
  - Xây dựng module Hybrid Search (BM25 + Dense Qdrant) và kiểm thử độ phủ từ khóa pháp lý.
- **Tuần 2:**
  - Tích hợp Cross-Encoder Reranker (`bge-reranker-v2-m3`) và Contextual Prepending.
  - Xây dựng pipeline tự động đánh giá RAGAS và tối ưu prompt generation để đạt Faithfulness >= 0.90.
