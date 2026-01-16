# ĐỀ CƯƠNG CHI TIẾT DỰ ÁN PHẦN MỀM
**Tên đề tài:** Hệ thống Hỗ trợ Đánh giá Mức độ Tập trung và Cảm xúc trong Lớp học Thông minh (Teacher Station Pro)

---

## 1. TỔNG QUAN DỰ ÁN (PROJECT OVERVIEW)

### 1.1. Đặt vấn đề
Trong môi trường giáo dục hiện đại, đặc biệt là tại các lớp học thực hành hoặc giảng đường lớn, giảng viên gặp khó khăn trong việc nắm bắt trạng thái cảm xúc và mức độ tiếp thu của toàn bộ sinh viên theo thời gian thực.
*   **Hạn chế của phương pháp quan sát truyền thống:** Giảng viên không thể quan sát đồng thời biểu cảm của 30-50 sinh viên. Những dấu hiệu như sự chán nản, mất tập trung hay bối rối thường bị bỏ qua.
*   **Thiếu dữ liệu định lượng:** Việc đánh giá chất lượng buổi dạy thường dựa vào cảm tính hoặc các bài kiểm tra cuối giờ, thiếu các chỉ số đo lường liên tục trong quá trình giảng dạy để điều chỉnh phương pháp kịp thời.

### 1.2. Giải pháp đề xuất
Xây dựng **Teacher Station Pro** - một ứng dụng Desktop hoạt động độc lập (Standalone) hỗ trợ giảng viên. Hệ thống sử dụng Camera gắn tại lớp học và công nghệ Trí tuệ nhân tạo (AI/Deep Learning) để tự động quét, nhận diện khuôn mặt và phân tích cảm xúc của tập thể lớp. Kết quả được tổng hợp thành các chỉ số trực quan (Dashboard) giúp giảng viên điều chỉnh nhịp độ bài giảng ngay lập tức.

---

## 2. YÊU CẦU CHỨC NĂNG (FUNCTIONAL REQUIREMENTS)

### 2.1. Phân hệ Giám sát Thời gian thực (Real-time Monitoring)
*   **Kết nối Camera đa luồng:** Hỗ trợ kết nối với Webcam tích hợp hoặc Camera giám sát ngoài (USB/IP Camera) để bao quát toàn bộ lớp học.
*   **Nhận diện đa khuôn mặt:** Tự động phát hiện và định vị tất cả khuôn mặt sinh viên xuất hiện trong khung hình.
*   **Phân lớp Cảm xúc & Giới tính:**
    *   Xác định 7 trạng thái cảm xúc cơ bản: Vui, Buồn, Giận dữ, Ngạc nhiên, Sợ hãi, Ghê tởm, Bình thường.
    *   Nhận diện giới tính (Nam/Nữ) để phục vụ thống kê nhân khẩu học.
*   **Hiển thị trực quan (Visual Feedback):** Vẽ khung bao quanh khuôn mặt với mã màu tương ứng mức độ tích cực (Xanh: Tích cực - Đỏ: Tiêu cực) ngay trên luồng video.

### 2.2. Phân hệ Phân tích & Thống kê (Analytics Dashboard)
*   **Tính điểm Tập trung (Engagement Score):** Tính toán chỉ số tập trung trung bình của cả lớp theo thang điểm 0.0 - 1.0 dựa trên trọng số cảm xúc.
*   **Biểu đồ Diễn biến (Live Trend Chart):** Vẽ đồ thị đường thể hiện sự thay đổi của mức độ tập trung theo thời gian thực (Real-time). Giúp giảng viên nhận biết thời điểm nào trong bài giảng thu hút hoặc gây nhàm chán.
*   **Thống kê Tổng hợp:**
    *   Biểu đồ tròn (Pie Chart) tỷ lệ Nam/Nữ.
    *   Biểu đồ cột (Bar Chart) phân bố các loại cảm xúc chủ đạo (Ví dụ: 60% Bình thường, 20% Vui, 20% Buồn).

### 2.3. Phân hệ Quản trị Dữ liệu (Data Management)
*   **Lưu trữ Phiên làm việc (Session Logging):** Tự động lưu toàn bộ dữ liệu thống kê của buổi học vào Cơ sở dữ liệu cục bộ (Local Database).
*   **Xem lại Lịch sử (History Review):** Giảng viên có thể xem lại biểu đồ và các chỉ số của các buổi học trước đó để so sánh hiệu quả giảng dạy.
*   **Xuất báo cáo:** Hỗ trợ xuất dữ liệu thô ra định dạng CSV/Excel để phục vụ nghiên cứu sâu hơn.

---

## 3. YÊU CẦU PHI CHỨC NĂNG (NON-FUNCTIONAL REQUIREMENTS)
*   **Tính Riêng tư (Privacy First):** Hệ thống chỉ xử lý và lưu trữ các **chỉ số thống kê** (vector cảm xúc, điểm số). **Không** ghi lại hay lưu trữ video/hình ảnh khuôn mặt của sinh viên, đảm bảo tuân thủ quy định về quyền riêng tư.
*   **Hiệu năng:** Tốc độ xử lý đạt tối thiểu 15-20 FPS trên máy tính cấu hình phổ thông (Core i5, không cần GPU rời) để đảm bảo tính thời gian thực.
*   **Tính Sẵn sàng:** Ứng dụng hoạt động Offline hoàn toàn, không phụ thuộc vào kết nối Internet.

---

## 4. CÔNG NGHỆ VÀ CÔNG CỤ (TECH STACK)

| Hạng mục | Công nghệ / Công cụ | Lý do lựa chọn |
| :--- | :--- | :--- |
| **Ngôn ngữ** | Python 3.9+ | Hệ sinh thái AI/Data Science mạnh mẽ, hỗ trợ tốt các thư viện xử lý ảnh. |
| **Giao diện (GUI)** | CustomTkinter | Thư viện giao diện người dùng hiện đại (Modern UI), hỗ trợ Dark/Light mode, vượt trội so với Tkinter truyền thống. |
| **Computer Vision** | OpenCV | Thư viện chuẩn công nghiệp cho xử lý ảnh đầu vào và thao tác trên khung hình. |
| **AI Model (Emotion)** | **PyTorch (ResNet18)** | Kiến trúc CNN mạnh mẽ (Deep Residual Learning), được huấn luyện lại (Fine-tuned) trên bộ dữ liệu FER2013 cho độ chính xác cao. |
| **AI Model (Gender)** | **Keras/TensorFlow** | Sử dụng kiến trúc Mini_XCEPTION tối ưu hóa, nhẹ và nhanh cho tác vụ phân loại nhị phân. |
| **Database** | SQLite | Cơ sở dữ liệu quan hệ nhúng, không cần cài đặt Server riêng (Zero-configuration), phù hợp cho ứng dụng Desktop. |
| **Đóng gói** | PyInstaller | Đóng gói toàn bộ môi trường Python và thư viện thành một file thực thi `.exe` duy nhất. |

---

## 5. THUẬT TOÁN CỐT LÕI (CORE ALGORITHM)

### 5.1. Quy trình Xử lý (Pipeline)
1.  **Input:** Nhận luồng Video từ Camera ($Frame_t$).
2.  **Face Detection:** Sử dụng Haar Cascade hoặc Face Detector để xác định vùng khuôn mặt ($ROI_{face}$).
3.  **Preprocessing:** Chuyển ảnh xám (Grayscale), chuẩn hóa kích thước (48x48 cho Emotion, 64x64 cho Gender), chuẩn hóa điểm ảnh [0,1].
4.  **Inference:** Đưa ảnh đã xử lý vào 2 Model song song (EmotionNet & GenderNet).
5.  **Scoring:** Tính điểm tập trung và cập nhật Dashboard.

### 5.2. Công thức tính Chỉ số Tập trung (Engagement Score)
Chỉ số tập trung $E(t)$ tại thời điểm $t$ được tính dựa trên trọng số của các nhóm cảm xúc phát hiện được:

$$ E(t) = \frac{\sum_{i=1}^{N} W(e_i)}{N} $$

Trong đó:
*   $N$: Tổng số khuôn mặt phát hiện được.
*   $e_i$: Cảm xúc của sinh viên thứ $i$.
*   $W(e)$: Trọng số quy đổi của cảm xúc:
    *   **Tích cực (Happy, Surprise):** $W = 1.0$ (Biểu hiện sự hứng thú, tiếp thu tốt).
    *   **Trung tính (Neutral):** $W = 0.5$ (Đang lắng nghe, mức độ tập trung cơ bản).
    *   **Tiêu cực (Sad, Angry, Fear, Disgust):** $W = 0.1$ (Có dấu hiệu chán nản, không hiểu bài hoặc xao nhãng).

### 5.3. Kỹ thuật Làm mượt (Smoothing)
Để tránh biểu đồ bị dao động quá mạnh do nhiễu, áp dụng kỹ thuật Trung bình trượt số mũ (Exponential Moving Average):
$$ S_{display}(t) = \alpha \cdot E(t) + (1 - \alpha) \cdot S_{display}(t-1) $$
*(Với $\alpha = 0.2$ giúp biểu đồ mượt mà hơn)*.

---

## 6. LỘ TRÌNH THỰC HIỆN (TIMELINE)
Dự án được chia làm 4 giai đoạn phát triển:

*   **Phase 1: Nghiên cứu & Xây dựng Core AI**
    *   Thu thập dữ liệu, huấn luyện Model Emotion (ResNet18) và Gender.
    *   Đánh giá độ chính xác Model (Accuracy > 65% trên tập Test FER2013).
*   **Phase 2: Phát triển Backend & Logic**
    *   Xây dựng module `FaceDetector`, `EngagementCalculator`.
    *   Thiết kế CSDL SQLite và các hàm CRUD.
*   **Phase 3: Phát triển Giao diện (GUI Integration)**
    *   Thiết kế giao diện Dashboard với CustomTkinter.
    *   Tích hợp luồng Video và vẽ biểu đồ Matplotlib thời gian thực.
*   **Phase 4: Kiểm thử & Đóng gói**
    *   Kiểm thử tích hợp (Integration Test).
    *   Tối ưu hóa hiệu năng (Threading).
    *   Đóng gói sản phẩm (.exe) và viết tài liệu hướng dẫn.
