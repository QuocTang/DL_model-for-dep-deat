# CHƯƠNG 1. TỔNG QUAN

## 1.1. Giới thiệu đề tài

Giới thiệu về bài toán nhận dạng và phân loại bệnh trên lá cây sầu riêng bằng phương pháp Machine Learning, tầm quan trọng của việc ứng dụng ML trong nông nghiệp thông minh để phát hiện sớm bệnh hại cây trồng.

## 1.2. Mục tiêu đề tài

Xây dựng mô hình Machine Learning (cụ thể là mô hình YOLO – thuộc nhánh Deep Learning trong ML) để phát hiện và phân loại các loại bệnh trên lá sầu riêng từ hình ảnh (bệnh cháy lá, bệnh thán thư, bệnh đốm mắt cua, sâu ăn, lá khỏe bình thường…).

## 1.3. Phạm vi & Giới hạn đề tài

Tập trung xử lý ảnh lá sầu riêng với tập dữ liệu gồm 15 lớp bệnh/trạng thái ban đầu, sau đó gom nhóm thành 6 lớp chính. Sử dụng mô hình pretrained YOLOv11x và fine-tune trên dữ liệu thực tế.

---

# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT

_(Chương này trình bày lý thuyết nền tảng liên quan đến đề tài)_

## 2.1. Tổng quan về Machine Learning

- **2.1.1.** Định nghĩa Machine Learning và các loại học máy: Supervised, Unsupervised, Reinforcement Learning.
- **2.1.2.** Quy trình xây dựng một hệ thống ML: Thu thập dữ liệu → Tiền xử lý → Huấn luyện → Đánh giá → Triển khai.
- **2.1.3.** Vị trí của Deep Learning trong hệ sinh thái Machine Learning.

## 2.2. Mạng nơ-ron và xử lý ảnh

- **2.2.1.** Khái niệm mạng nơ-ron nhân tạo (Artificial Neural Networks).
- **2.2.2.** Mạng nơ-ron tích chập (Convolutional Neural Networks - CNN) và vai trò trong bài toán phân loại/phát hiện ảnh.

## 2.3. Bài toán phát hiện đối tượng (Object Detection)

- **2.3.1.** Định nghĩa bài toán phát hiện đối tượng trong ảnh.
- **2.3.2.** Phân loại các phương pháp: Two-stage (R-CNN, Faster R-CNN) vs One-stage (YOLO, SSD).

## 2.4. Kiến trúc YOLO (You Only Look Once)

- **2.4.1.** Nguyên lý hoạt động của YOLO: chia lưới, dự đoán bounding box và class probability.
- **2.4.2.** Sự phát triển từ YOLOv1 đến YOLOv11 và các cải tiến chính.
- **2.4.3.** Kiến trúc YOLOv11x: Backbone, Neck, Head và các đặc điểm nổi bật.

## 2.5. Transfer Learning & Fine-tuning

- **2.5.1.** Khái niệm Transfer Learning và lợi ích khi áp dụng cho tập dữ liệu nhỏ.
- **2.5.2.** Chiến lược fine-tuning mô hình pretrained trên dữ liệu mới.

## 2.6. Các kỹ thuật xử lý mất cân bằng dữ liệu (Data Imbalance)

- **2.6.1.** Vấn đề mất cân bằng lớp trong bài toán phân loại/phát hiện.
- **2.6.2.** Các phương pháp: Oversampling, Undersampling (Downsampling), Data Augmentation.

## 2.7. Các chỉ số đánh giá mô hình

- **2.7.1.** Precision, Recall, F1-Score.
- **2.7.2.** mAP (mean Average Precision) – chỉ số chuẩn cho bài toán Object Detection.
- **2.7.3.** Confusion Matrix.

---

# CHƯƠNG 3. NỘI DUNG TIỂU LUẬN (CÁCH TIẾP CẬN VÀ PHƯƠNG PHÁP XÂY DỰNG)

_(Chương này mô tả chi tiết cách tiếp cận và các bước xây dựng mô hình)_

## 3.1. Quy trình tổng thể

Sơ đồ pipeline từ thu thập dữ liệu → khảo sát/phân tích → tiền xử lý → huấn luyện → đánh giá mô hình.

## 3.2. Thu thập dữ liệu

- **3.2.1. Nguồn dữ liệu:** Sử dụng Roboflow API để tải dataset "durian" (workspace: `cico-siefo`, project: `durian-k51j3`, version 1).
- **3.2.2. Cấu trúc dữ liệu:** Định dạng YOLOv11 gồm thư mục `images/` và `labels/` cho mỗi split (train/valid/test).
- **3.2.3. Mô tả các lớp bệnh ban đầu (15 lớp):** Bệnh bọ trĩ, bệnh phấn trắng, bệnh cháy lá, bệnh đốm mắt cua, bệnh gỉ sắt, bệnh lá vàng, bệnh tảo đỏ, bệnh thán thư, đốm lá do nấm, đốm sinh lý nhẹ, lá khỏe bình thường, sâu vẽ bùa, sâu ăn, vàng sinh lý, vàng thiếu magie.

## 3.3. Phân tích và Khảo sát dữ liệu (EDA)

- **3.3.1.** Trực quan hóa phân bố lớp (Class Distribution) trên tập train và valid bằng biểu đồ cột (Seaborn barplot).
- **3.3.2.** Nhận xét về sự mất cân bằng dữ liệu giữa các lớp.

## 3.4. Tiền xử lý dữ liệu

- **3.4.1. Gom nhóm lớp (Refactoring):** Gom 15 lớp ban đầu thành 6 lớp chính dựa trên tần suất và ý nghĩa thực tiễn:
  | Lớp mới | Tên bệnh | ID gốc |
  |---------|-----------|--------|
  | 0 | Sâu ăn | 12 |
  | 1 | Bệnh cháy lá | 2 |
  | 2 | Bệnh thán thư | 7 |
  | 3 | Bệnh đốm mắt cua | 3 |
  | 4 | Lá khỏe bình thường | 10 |
  | 5 | Nhóm các bệnh còn lại (Other) | Còn lại |
- **3.4.2. Cân bằng dữ liệu (Downsampling):** Giảm mẫu các lớp chiếm ưu thế (bệnh cháy lá, sâu ăn) xuống còn 40% để giảm thiên lệch (bias).
- **3.4.3.** Sao chép ảnh và viết lại file label theo mapping ID mới.

## 3.5. Huấn luyện mô hình

- **3.5.1. Mô hình sử dụng:** YOLOv11x pretrained (`yolo11x.pt`).
- **3.5.2. Cấu hình huấn luyện:**

  | Tham số                 | Giá trị          |
  | ----------------------- | ---------------- |
  | Batch size              | 8                |
  | Epochs                  | 100              |
  | Image size              | 640×640          |
  | Optimizer               | AdamW            |
  | Learning rate (lr0)     | 0.005            |
  | LR final (lrf)          | 0.10             |
  | Weight decay            | 0.0005           |
  | Warmup epochs           | 3                |
  | LR scheduler            | Cosine Annealing |
  | Early Stopping patience | 50               |
  | Seed                    | 42               |

- **3.5.3. Phần cứng:** Sử dụng GPU CUDA (`device=0`).

---

# CHƯƠNG 4. CÀI ĐẶT & THỰC NGHIỆM

_(Chương này trình bày minh họa & demo về kết quả)_

## 4.1. Môi trường và Công cụ phát triển

| Thành phần      | Chi tiết                                  |
| --------------- | ----------------------------------------- |
| Ngôn ngữ        | Python                                    |
| Framework ML    | Ultralytics (YOLOv11), PyTorch            |
| Thư viện hỗ trợ | Roboflow SDK, Pandas, Seaborn, Matplotlib |
| Phần cứng       | GPU NVIDIA hỗ trợ CUDA                    |

## 4.2. Kịch bản thực nghiệm & Kết quả

- **4.2.1. Kết quả huấn luyện:** Biểu đồ loss (train/val), learning rate qua từng epoch.
- **4.2.2. Kết quả đánh giá trên tập validation:** Precision, Recall, mAP@0.5, mAP@0.5:0.95.
- **4.2.3. Confusion Matrix:** Ma trận nhầm lẫn giữa các lớp bệnh.

## 4.3. Minh họa kết quả phát hiện (Demo)

- **4.3.1. Hình ảnh inference:** Ảnh đầu vào và kết quả phát hiện bệnh (bounding box + nhãn + confidence score).
- **4.3.2. Phân tích:** Các trường hợp phát hiện đúng/sai tiêu biểu.

## 4.4. Đánh giá hiệu năng

So sánh với baseline, nhận xét về tốc độ inference và độ chính xác trên từng lớp bệnh.

---

# CHƯƠNG 5. KẾT LUẬN & HƯỚNG PHÁT TRIỂN

## 5.1. Kết quả đạt được

Tóm tắt mô hình đã đạt được những chỉ số đánh giá nào, khả năng phát hiện bệnh trên lá sầu riêng trong thực tế.

## 5.2. Hạn chế

Các điểm yếu (ví dụ: dữ liệu chưa đủ đa dạng, một số lớp bệnh có quá ít mẫu, chưa triển khai thành ứng dụng thực tế).

## 5.3. Hướng phát triển

- Mở rộng tập dữ liệu với nhiều loại bệnh và điều kiện chụp ảnh khác nhau.
- Áp dụng Data Augmentation nâng cao (Mosaic, MixUp).
- Triển khai mô hình trên thiết bị di động/edge device phục vụ nông dân.
- Tích hợp vào hệ thống giám sát sức khỏe vườn cây tự động.

---

# TÀI LIỆU THAM KHẢO

Liệt kê các tài liệu, bài báo khoa học, link GitHub, và tài nguyên đã tham khảo:

- Bài báo gốc về kiến trúc YOLO (Joseph Redmon et al.)
- Tài liệu Ultralytics YOLOv11
- Roboflow Documentation
- Các bài báo về ứng dụng Machine Learning trong phát hiện bệnh cây trồng
- PyTorch Documentation
