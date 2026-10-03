# Đánh giá benchmark hệ thống memory cho agent

## Kết quả benchmark

### Standard benchmark (10 phiên hội thoại)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline Agent | 4,047 | 17,498 | 0.0% | 20.0% | 0 | 0 |
| Advanced Agent | 5,819 | 29,077 | 100.0% | 100.0% | 281 | 0 |

### Long-context stress benchmark (hội thoại dài 16 lượt)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline Agent | 2,868 | 23,520 | 0.0% | 20.0% | 0 | 0 |
| Advanced Agent | 4,039 | 10,817 | 100.0% | 100.0% | 217 | 12 |

## Phân tích kết quả

### Khả năng nhớ thông tin giữa các phiên
Baseline Agent chỉ lưu tin nhắn trong từng thread. Khi sang thread mới để trả lời câu hỏi kiểm tra, agent không còn context cũ nên recall đạt 0%.

Advanced Agent lưu thông tin người dùng (tên, nơi ở, nghề nghiệp, món ăn ưa thích, phong cách trả lời) vào `state/profiles/<user>/User.md`. Khi mở thread mới, agent nạp file này vào prompt để trả lời, nhờ đó giữ được thông tin giữa các phiên và đạt recall 100%.

### Chi phí token ở hội thoại ngắn
Trong bài test standard gồm 10 phiên ngắn, Advanced Agent dùng 29,077 prompt tokens, cao hơn mức 17,498 của Baseline.

Lý do là Advanced Agent luôn gắn nội dung `User.md` vào đầu prompt ở mỗi lượt gọi. Với các hội thoại ngắn dưới 10 lượt, lượng token của file profile làm tăng tổng token đầu vào.

### Cơ chế compact memory ở hội thoại dài
Trong bài stress test 16 lượt, tương quan token đảo ngược:
- Baseline giữ nguyên văn toàn bộ các đoạn tin nhắn dài qua từng lượt, làm lượng prompt token tích lũy lên 23,520.
- Advanced Agent kích hoạt compact memory 12 lần khi tổng token trong thread vượt ngưỡng 800. Hệ thống nén các tin nhắn cũ thành bản tóm tắt ngắn và chỉ giữ 4 tin nhắn gần nhất trong prompt. Nhờ vậy, prompt token xử lý giảm còn 10,817, thấp hơn 54% so với Baseline.

### Tăng trưởng bộ nhớ và rủi ro vận hành
File `User.md` tăng khoảng 200 đến 300 bytes mỗi user do chỉ lưu các thuộc tính chọn lọc, không lưu toàn bộ lịch sử chat.

Việc lưu profile dạng markdown có hai rủi ro chính:
- Nhiễm bẩn dữ liệu: các câu nói đùa, ví dụ giả định hoặc câu hỏi tu từ có thể bị trích xuất nhầm thành thông tin cá nhân nếu thiếu bộ lọc.
- Xung đột dữ liệu: khi người dùng thay đổi nơi ở hoặc công việc, hệ thống cần ghi đè giá trị cũ thay vì ghi thêm, tránh để profile chứa dữ liệu mâu thuẫn.

## Cải tiến kỹ thuật mở rộng (Bonus)

Hệ thống đã bổ sung cơ chế trích xuất thực thể có cấu trúc kết hợp xử lý xung đột (conflict resolution), khử nhiễu (distractor rejection) và lọc câu hỏi (query filtering).

### Vấn đề giải quyết
- Nhiễm bẩn bộ nhớ: Trong hội thoại tự nhiên, người dùng thường nói đùa (như nhắc đến việc làm Product Manager), nói về các chuyến đi ngắn hạn (như bay ra Hà Nội họp), hoặc đặt câu hỏi. Nếu trích xuất thô, những dữ liệu này sẽ bị ghi nhầm vào hồ sơ cá nhân.
- Dữ liệu lỗi thời và mâu thuẫn: Khi người dùng đổi nơi ở (từ Huế sang Đà Nẵng) hoặc đổi chuyên môn (từ backend sang MLOps), hệ thống cần cập nhật giá trị mới nhất và xóa bỏ giá trị cũ, tránh lưu đồng thời hai thông tin đối nghịch.

### Tác động đến recall và token cost
- Cải thiện recall: Đảm bảo agent truy xuất đúng thông tin hiện tại thay vì bị nhiễu bởi dữ liệu cũ, đạt 100% recall trong benchmark.
- Tối ưu chi phí token: Bộ lọc ngăn chặn các fact rác ghi vào `User.md`. File profile duy trì kích thước gọn gàng (200 đến 300 bytes), hạn chế phát sinh token đính kèm không cần thiết vào prompt.

### Rủi ro phát sinh
- Nguy cơ lọc nhầm (false negative): Nếu quy tắc lọc quá chặt, hệ thống có thể bỏ sót thông tin thực khi người dùng diễn đạt khác mẫu (ví dụ: người dùng thực sự chuyển việc sang Product Manager nhưng nói vắn tắt).
- Mất lịch sử cập nhật: Cơ chế ghi đè trực tiếp giúp xử lý xung đột tức thời nhưng làm mất lịch sử thay đổi (audit trail), khiến hệ thống không thể truy vết lại dữ liệu cũ khi cần đối chiếu.

