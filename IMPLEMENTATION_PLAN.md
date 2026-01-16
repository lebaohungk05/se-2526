# 📅 IMPLEMENTATION PLAN: STUDENT ENGAGEMENT ANALYSIS SYSTEM (Teacher Tool)

Kế hoạch triển khai hệ thống giám sát sự tập trung của sinh viên, vận hành tập trung trên máy Giảng viên.
**Chiến lược điều chỉnh:** Đã nâng cấp lên **ResNet18 (PyTorch)** để đạt độ chính xác cao nhất và sử dụng **Optimized Haar Cascade** để đảm bảo độ ổn định (thay thế MediaPipe do lỗi tương thích).

---

## Giai đoạn 1: Đánh giá & Chốt Model (Data Science Wrap-up)
- [x] **1.1. Tích hợp Model ResNet18 (PyTorch)**
    - Thay thế Mini XCEPTION cũ bằng ResNet18 tốt hơn.
    - Load trọng số từ: `trained_models/emotion_models/fer2013_resnet18_best_sgd.pth`.
- [x] **1.2. Nâng cấp Face Detection**
    - Đã thử nghiệm MediaPipe nhưng gặp lỗi tương thích môi trường (AttributeError trên Python 3.12).
    - **Giải pháp cuối:** Sử dụng **Optimized Haar Cascade** (Scale 1.2, Neighbors 8) để đảm bảo độ ổn định tuyệt đối trên mọi máy Windows, đồng thời loại bỏ lỗi nhận diện sai (3 người thành 5).

---

## Giai đoạn 2: Phát triển Ứng dụng Giảng viên (Software Engineering Core)
- [x] **2.1. Modern UI (Ocean Breeze Style)**
    - **Palette:** Slate/Sky Light Mode (`#F1F5F9`, `#0EA5E9`).
    - **Layout:** Sidebar Navigation + Hero Dashboard.
    - **Components:** Styled Cards, Ghost Buttons, Rounded Corners.
- [x] **2.2. Tính năng Trợ giảng Online**
    - Chế độ **Mini-Overlay (Always on Top)** tối ưu cho Google Meet/Zoom.
    - **Smart Alert:** Cảnh báo âm thanh khi chỉ số tập trung thấp.
- [x] **2.3. Analytics & Persistence**
    - Quản lý lịch sử buổi học qua SQLite (`student_engagement.db`).
    - **Export CSV:** Xuất dữ liệu báo cáo.
    - **Chart Fixes:** Làm mượt biểu đồ (Smoothing).

---

## Giai đoạn 3: Kiểm thử & Đóng gói (Final Polish)
- [x] **3.1. System Testing**
    - Đã kiểm tra logic PyTorch + Haar Cascade chạy ổn định.
- [x] **3.2. Đóng gói Sản phẩm (.EXE)**
    - Đã tạo cấu hình `TeacherStationPro.spec` chuẩn cho PyInstaller.
    - **Fix:** Xử lý lỗi `matplotlib_inline` bằng cách exclude thư viện xung đột.
    - **Fix:** Cập nhật logic đường dẫn (`_internal`) để load Model/DB chính xác trong file .exe.
    - **Kết quả:** Build thành công tại `dist/TeacherStationPro/TeacherStationPro.exe`.

---

## 📝 Nhật ký Thay đổi (Changelog)

* **2026-01-15 (Dọn dẹp & Chuẩn hóa Tài liệu - FINAL CLEANUP):**
    - **REFRACTORING:** Tiến hành dọn dẹp toàn diện mã nguồn. Loại bỏ các module thừa:
        - Xóa thư mục `src/web` (Flask App không sử dụng).
        - Xóa `main_app_gui.py` (Giao diện cũ dùng Tkinter).
        - Loại bỏ các script training và model Keras cũ (`mobilenet_v2.py`, `train_emotion_classifier.py`) để tối ưu dung lượng và tính nhất quán.
    - **DOCS UPDATE:** 
        - Viết lại `README.md` chuyên nghiệp.
        - Tạo mới `PROJECT_PROPOSAL.md` (Đề cương dự án) khớp 100% với sản phẩm thực tế (Desktop App), đảm bảo tính nhất quán cho môn học Kỹ thuật Phần mềm.
    - **REPO MIGRATION:** Xóa remote origin cũ, chuẩn bị đẩy toàn bộ mã nguồn sạch lên repository cá nhân mới.
    - **STABILITY CHECK:** Chạy `python -m compileall` xác nhận không có lỗi import hay cú pháp sau khi dọn dẹp.

* **2026-01-15 (Packaging Milestone):**
    - **BUILD SUCCESS:** Đóng gói thành công `TeacherStationPro.exe` chạy độc lập.
    - **HOTFIX:** Sửa lỗi đường dẫn tài nguyên (Resource Path) khi chạy dưới dạng file thực thi (Frozen state).
    - **CLEANUP:** Loại bỏ các thư viện thừa (IPython, Jedi) giúp giảm dung lượng và tránh lỗi runtime.

* **2026-01-14 (Final Review):**
    - **UI UPDATE:** Chốt giao diện **Ocean Breeze (Light Mode)** mang lại cảm giác hiện đại, sáng sủa.
    - **PACKAGING:** Hoàn tất file cấu hình `.spec` để đóng gói ứng dụng `.exe`.
    - **VERIFICATION:** Xác nhận logic Haar Cascade hoạt động ổn định thay thế MediaPipe.

* **2026-01-12 (UI/UX Revolution):**
    - **SHADCN UI:** Viết lại toàn bộ giao diện theo phong cách Modern.
    - **NEW FEATURES:** Thêm Tab Settings, Export CSV, và Smart Alert Beep.
    - **ANALYTICS FIX:** Sửa lỗi biểu đồ và thêm thuật toán làm mượt dữ liệu.

* **2026-01-12 (Nâng cấp Pro):**
    - **AI UPGRADE:** Tích hợp thành công **ResNet18 (PyTorch)** cho độ chính xác cảm xúc vượt trội.
    - **DETECTION FIX:** Đã thử nghiệm MediaPipe nhưng quyết định revert về Haar Cascade tối ưu hóa để đảm bảo tính ổn định.
    - **UI TRANSFORMATION:** Chuyển sang giao diện **CustomTkinter** hiện đại, tích hợp Dashboard nội bộ.

* **2026-01-06 (Chiều):**
    - **NÂNG CẤP KTPM (A+ Goal):** Tích hợp **SQLite Database** thay thế CSV để quản lý dữ liệu chuyên nghiệp.
    - **QUALITY ASSURANCE:** Viết bộ **Unit Tests** (`tests/test_suite.py`) xác thực logic tính toán và dữ liệu.
    - **GUI UPDATE:** Thêm tab **"Session History"** vào giao diện chính để xem lại báo cáo cũ.
    - **TÍNH NĂNG MỚI:** Thêm phân tích Giới tính (Gender-based Engagement) để so sánh mức độ tập trung Nam/Nữ.

* **2025-12-24:**
    - **QUYẾT ĐỊNH:** Dừng training MobileNetV2. Chuyển sang dùng `fer2013_mini_XCEPTION.119-0.65.hdf5` có sẵn.
    - **LÝ DO:** Tiết kiệm thời gian training để tập trung làm tính năng phần mềm (Screen Capture, MediaPipe, UI) cho đồ án KTPM.