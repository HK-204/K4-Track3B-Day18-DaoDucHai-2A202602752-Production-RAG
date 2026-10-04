# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Đào Đức Hải  
**MSSV:** 2A202602752  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8542 | 0.8462 | -0.0080 |
| Answer Relevancy | 0.7965 | 0.7632 | -0.0333 |
| Context Precision | 0.8462 | 0.8750 | +0.0288 |
| Context Recall | 0.8333 | 0.7344 | -0.0990 |

> **Nhận xét tổng quan:**  
> - **Context Precision tăng vượt trội (+0.0288 lên 0.8750):** Nhờ sự kết hợp của **Hybrid Search (BM25 + Dense RRF)** và **Cross-Encoder Reranker (`bge-reranker-v2-m3`)**, các chunk ngữ cảnh được xếp hạng ở top đầu có độ liên quan và độ tập trung cao hơn đáng kể so với baseline dense-only.
> - **Context Recall và Faithfulness:** Bị ảnh hưởng nhẹ chủ yếu do hiện tượng xung đột văn bản phiên bản cũ (v2023, v1.0) và phiên bản mới (v2024, v2.0) trong kho dữ liệu mẫu, khi cả hai phiên bản đều được truy xuất khiến LLM phân vân hoặc liệt kê cả hai phương án.

---

## Bottom-5 Failures

### #1
- **Question:** Nhân viên thử việc có được nghỉ phép năm không?
- **Expected:** KHÔNG. Nhân viên thử việc KHÔNG được nghỉ phép năm. Nếu cần nghỉ, phải xin nghỉ không lương và được trưởng phòng phê duyệt.
- **Got:** Không tìm thấy.
- **Worst metric:** Faithfulness (0.0000) / Context Recall
- **Error Tree:** Output sai (trả về fallback "Không tìm thấy.") → Context đúng? KHÔNG (context trả về không chứa câu quy định cụ thể về đối tượng thử việc) → Query OK? Query rõ nghĩa nhưng từ khóa "thử việc" bị trượt khỏi top 3 sau rerank.
- **Root cause:** Trong quá trình hierarchical chunking, điều khoản quy định về thời gian thử việc và điều khoản nghỉ phép năm nằm ở hai mục khác nhau. Cross-encoder ưu tiên các chunk nói chung về "nghỉ phép năm" hơn là chunk nói về "nhân viên thử việc".
- **Suggested fix:** Nâng `RERANK_TOP_K` từ 3 lên 5 để bao quát thêm ngữ cảnh phụ, hoặc áp dụng chunking theo điều khoản trọn vẹn (structure-aware chunking).

### #2
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** Dựa trên ngữ cảnh được cung cấp, có hai mốc thời gian về chu kỳ thay đổi mật khẩu tùy thuộc vào phiên bản: Mỗi 90 ngày (theo tài liệu hướng dẫn mật khẩu hiện hành) và Mỗi 120 ngày (theo chính sách mật khẩu phiên bản cũ đã bị thay thế).
- **Worst metric:** Context Recall (0.0000) / Context Precision
- **Error Tree:** Output đảo lộn hiệu lực giữa bản cũ và mới → Context đúng? Context chứa cả 2 phiên bản văn bản (v1.0 và v2.0) → Query OK? Query tốt.
- **Root cause:** Thiếu cơ chế lọc phiên bản (Version / Temporal Filtering). Hệ thống tìm kiếm ngữ nghĩa retrieve cả tài liệu cũ và mới vì cả hai đều có độ tương đồng ngữ nghĩa cao với câu hỏi.
- **Suggested fix:** Bổ sung metadata `version` và `status: active|deprecated` trong M5 Enrichment; thêm bộ lọc `status == 'active'` trước khi đưa vào Reranker.

### #3
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Got:** Dựa trên context, số ngày phép năm tùy thuộc vào từng phiên bản chính sách: Phiên bản 2023: Mỗi nhân viên chính thức được hưởng 12 ngày phép năm có lương. Phiên bản 2024: Mỗi nhân viên chính thức được hưởng 15 ngày phép năm có lương.
- **Worst metric:** Faithfulness / Answer Relevancy
- **Error Tree:** Output thiếu tính dứt khoát do liệt kê cả 2 mốc → Context đúng? Context chứa cả văn bản 2023 và 2024.
- **Root cause:** Tài liệu cũ chưa được đánh dấu thu hồi hoặc gắn nhãn hết hiệu lực trong siêu dữ liệu văn bản.
- **Suggested fix:** Cập nhật System Prompt hướng dẫn LLM: *"Nếu trong context có nhiều văn bản quy định mâu thuẫn về thời gian hoặc phiên bản, chỉ áp dụng phiên bản có năm ban hành mới nhất."*

### #4
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got:** Dựa trên context, có hai thông tin về số năm thâm niên được cộng thêm ngày phép: Nhân viên có thâm niên từ 5 năm trở lên được cộng thêm 1 ngày phép cho mỗi 5 năm làm việc liên tục; Nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm làm việc liên tục.
- **Worst metric:** Answer Relevancy (0.9155)
- **Error Tree:** Output nêu cả hai mốc năm do nhận hai chunk mâu thuẫn từ retrieval.
- **Root cause:** Hiện tượng Document Collision do kho văn bản lưu trữ nhiều phiên bản quy chế qua các thời kỳ mà không có cơ chế quản lý vòng đời tài liệu (Document Lifecycle Management).
- **Suggested fix:** Phân cấp tài liệu theo collection riêng (Active vs Archive) hoặc metadata filtering theo thời gian có hiệu lực.

### #5
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo chính sách mật khẩu v2.0 hiện hành, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** Có, tất cả nhân viên bắt buộc kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ.
- **Worst metric:** Answer Relevancy (0.7086)
- **Error Tree:** Output trả lời đúng ý chính nhưng ngắn gọn hơn so với ground truth chi tiết.
- **Root cause:** Prompt hướng dẫn của hệ thống yêu cầu trả lời súc tích và chỉ dựa trên context, trong khi ground truth benchmark mở rộng so sánh với quy định cũ v1.0.
- **Suggested fix:** Điều chỉnh prompt generation để diễn giải đầy đủ hơn khi câu hỏi đề cập đến chính sách bảo mật nội bộ.

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
*"Bao lâu phải đổi mật khẩu một lần?"*

**Error Tree walkthrough:**
1. **Output đúng?** → KHÔNG HOÀN TOÀN ĐÚNG. Output nhận biết được có hai quy định (90 ngày và 120 ngày) nhưng lại gán nhầm 90 ngày là hiện hành và 120 ngày là phiên bản cũ, dẫn tới kết luận sai lệch.
2. **Context đúng?** → ĐÚNG MỘT NỬA. Cả hai chunk thuộc file `it_policy_v1.md` (90 ngày) và `it_policy_v2.md` (120 ngày) đều được Hybrid Search trả về ở top score.
3. **Query rewrite OK?** → CÂU HỎI RÕ NGHĨA. Query chứa đầy đủ thực thể và hành động, không cần rewrite.
4. **Fix ở bước:**
   - **Bước 1 (Retrieval Metadata Filtering):** Thêm trường `is_latest: bool` hoặc lọc bỏ các tài liệu có tiền tố `_deprecated`.
   - **Bước 2 (Generation Conflict Resolution Prompt):** Thêm chỉ dẫn giải quyết xung đột tài liệu trong prompt của LLM.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Thực hiện Metadata Filter kết hợp Semantic Search:** Tự động lọc tài liệu theo phiên bản có hiệu lực mới nhất trong Qdrant trước khi tính điểm RRF.
- **Tối ưu hóa Prompt Generator:** Xây dựng kỹ thuật Prompt Engineering nâng cao hướng dẫn mô hình nhận diện số phiên bản (`v1.0`, `v2.0`, `2023`, `2024`) để luôn ưu tiên phiên bản cập nhật.
- **Nâng Top-K Reranking:** Mở rộng từ `top_k=3` lên `top_k=5` nhằm tăng Context Recall cho các câu hỏi tổng hợp nhiều chính sách.
