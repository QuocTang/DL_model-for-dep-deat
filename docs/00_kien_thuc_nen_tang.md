# 🧠 Kiến Thức Nền Tảng — Đọc trước khi bắt đầu

> **Mục tiêu:** Hiểu các khái niệm cơ bản nhất để đọc hiểu code trong dự án này.  
> Nếu bạn đã biết Deep Learning, bạn có thể bỏ qua file này.

---

## 1. Trí Tuệ Nhân Tạo (AI) là gì?

Hãy tưởng tượng bạn muốn dạy một đứa bé phân biệt **con chó** và **con mèo**:

- Bạn cho bé xem hàng trăm ảnh chó và mèo
- Mỗi ảnh bạn nói: "Đây là chó", "Đây là mèo"
- Sau một thời gian, bé tự phân biệt được mà không cần bạn nói

**AI làm tương tự**, nhưng thay vì bộ não con người, nó dùng **máy tính + toán học**.

---

## 2. Machine Learning (ML) vs Deep Learning (DL)

```
AI (Trí tuệ nhân tạo)
 └── Machine Learning (Học máy)
      └── Deep Learning (Học sâu) ← dự án này dùng cái này
```

|                    | Machine Learning                    | Deep Learning                       |
| ------------------ | ----------------------------------- | ----------------------------------- |
| **Cách hoạt động** | Con người chọn đặc trưng (features) | Máy **tự tìm** đặc trưng            |
| **Ví dụ**          | "Chó có tai dài" → viết code đo tai | Cho xem 10.000 ảnh → máy tự nhận ra |
| **Khi nào dùng**   | Dữ liệu nhỏ, đơn giản               | Hình ảnh, âm thanh, văn bản         |
| **Cần gì**         | CPU đủ dùng                         | Cần **GPU** mạnh                    |

---

## 3. Bài toán Object Detection

Dự án này giải quyết bài toán **Object Detection** (Phát hiện đối tượng).

### 3 loại bài toán thị giác máy tính:

```
┌─────────────────────────────────────────────────────────┐
│  Image Classification (Phân loại ảnh)                   │
│  → Cả ảnh là "con chó" hay "con mèo"?                  │
│  → Output: 1 nhãn cho toàn ảnh                          │
│                                                         │
│  Object Detection (Phát hiện đối tượng) ← DỰ ÁN NÀY   │
│  → Ở đâu trong ảnh có chó? Ở đâu có mèo?              │
│  → Output: nhiều bounding box + nhãn                    │
│                                                         │
│  Image Segmentation (Phân vùng ảnh)                     │
│  → Đánh dấu từng pixel thuộc chó hay mèo               │
│  → Output: mặt nạ (mask) cho từng pixel                 │
└─────────────────────────────────────────────────────────┘
```

### Bounding Box là gì?

```
  ┌──────────────────────┐
  │  ┌─────────┐         │
  │  │  BỆNH   │ ← Bounding box
  │  │ CHÁY LÁ │         │
  │  └─────────┘         │
  │         ┌───────┐    │
  │         │ LÁ    │    │
  │         │ KHỎE  │    │
  │         └───────┘    │
  └──────────────────────┘
       Ảnh lá sầu riêng
```

Mỗi bounding box được mô tả bằng 5 số:

```
<class_id> <x_center> <y_center> <width> <height>
```

Ví dụ: `2 0.45 0.62 0.15 0.20`

- `2` = loại bệnh (bệnh cháy lá)
- `0.45` = tâm nằm ở 45% chiều rộng ảnh
- `0.62` = tâm nằm ở 62% chiều cao ảnh
- `0.15` = box rộng 15% ảnh
- `0.20` = box cao 20% ảnh

> Tất cả giá trị đều là tỷ lệ (0-1) so với kích thước ảnh, không phải pixel.

---

## 4. YOLO là gì?

**YOLO = You Only Look Once** (Bạn chỉ cần nhìn một lần)

### Tại sao tên "Nhìn một lần"?

Các model cũ phát hiện đối tượng theo 2 bước:

1. Quét toàn ảnh tìm "vùng nghi ngờ" có đối tượng
2. Phân loại từng vùng → rất chậm!

YOLO làm trong **1 bước duy nhất**:

- Chia ảnh thành lưới (grid)
- Mỗi ô lưới đồng thời dự đoán: có đối tượng không? Loại gì? Ở đâu?
- **Kết quả: nhanh hơn 100-1000x** so với model cũ

### Các phiên bản YOLO:

```
YOLOv1 (2016) → v2 → v3 → v4 → v5 → v6 → v7 → v8 → v9 → v10 → v11 (2024)
                                                                        ↑
                                                              Dự án này dùng v11x
```

### "x" trong YOLOv11x nghĩa là gì?

Mỗi phiên bản YOLO có nhiều kích cỡ:

| Variant           | Kích cỡ      | Tốc độ        | Độ chính xác | Khi nào dùng                              |
| ----------------- | ------------ | ------------- | ------------ | ----------------------------------------- |
| `n` (nano)        | Nhỏ nhất     | ⚡ Nhanh nhất | Thấp nhất    | Điện thoại, IoT                           |
| `s` (small)       | Nhỏ          | Nhanh         | Thấp         | Realtime trên CPU                         |
| `m` (medium)      | Trung bình   | TB            | TB           | Cân bằng                                  |
| `l` (large)       | Lớn          | Chậm          | Cao          | Server có GPU                             |
| `x` (extra large) | **Lớn nhất** | **Chậm nhất** | **Cao nhất** | **Khi cần chính xác nhất** ← chọn cái này |

---

## 5. GPU là gì? Tại sao cần GPU?

### CPU vs GPU

|              | CPU                          | GPU                                 |
| ------------ | ---------------------------- | ----------------------------------- |
| **Viết tắt** | Central Processing Unit      | Graphics Processing Unit            |
| **Ví von**   | 1 giáo sư giải toán rất giỏi | 1000 học sinh giải toán đồng thời   |
| **Giỏi về**  | Tác vụ phức tạp, tuần tự     | Tác vụ đơn giản, song song          |
| **DL cần**   | ❌ Quá chậm                  | ✅ Phù hợp (nhân ma trận song song) |

### CUDA là gì?

CUDA là **phần mềm do NVIDIA tạo ra** cho phép lập trình viên dùng GPU NVIDIA để tính toán.

- Chỉ GPU NVIDIA hỗ trợ CUDA
- PyTorch dùng CUDA để chạy DL trên GPU
- Không có CUDA → phải train trên CPU → **chậm 10-50 lần**

---

## 6. Quy trình training (huấn luyện) một model DL

```
       ┌─────────────┐
       │  DỮ LIỆU    │ ← 1. Thu thập & gán nhãn
       └──────┬───────┘
              │
       ┌──────▼───────┐
       │  TIỀN XỬ LÝ  │ ← 2. Làm sạch, cân bằng, augment
       └──────┬───────┘
              │
       ┌──────▼───────┐
       │  TRAINING     │ ← 3. Cho model "học" từ dữ liệu
       │  (Huấn luyện) │    (lặp đi lặp lại nhiều epoch)
       └──────┬───────┘
              │
       ┌──────▼───────┐
       │  VALIDATION   │ ← 4. Kiểm tra model trên dữ liệu mới
       └──────┬───────┘
              │
       ┌──────▼───────┐
       │  ĐÁNH GIÁ     │ ← 5. Đo các metrics (mAP, precision...)
       └──────┬───────┘
              │
       ┌──────▼───────┐
       │  TRIỂN KHAI    │ ← 6. Dùng model trong thực tế
       └──────────────┘
```

---

## 7. Thuật ngữ quan trọng

### Training (huấn luyện)

| Thuật ngữ             | Giải thích                                          | Ví von                        |
| --------------------- | --------------------------------------------------- | ----------------------------- |
| **Epoch**             | 1 lần model xem qua **toàn bộ** dữ liệu             | Đọc hết 1 cuốn sách           |
| **Batch**             | Nhóm ảnh xử lý cùng lúc (VD: 8 ảnh/lần)             | Giáo viên chấm 8 bài cùng lúc |
| **Iteration**         | 1 lần xử lý 1 batch                                 | Chấm xong 1 nhóm bài          |
| **Loss**              | Sai số — model sai bao nhiêu. **Càng nhỏ càng tốt** | Điểm trừ trong bài kiểm tra   |
| **Learning Rate**     | "Bước nhảy" mỗi lần cập nhật                        | Tốc độ học bài                |
| **Weight (trọng số)** | Các con số bên trong model = "kiến thức"            | Kinh nghiệm tích lũy          |
| **Gradient**          | Hướng cần thay đổi weight để giảm loss              | "Đáp án" cho biết sai ở đâu   |

### Đánh giá model

| Thuật ngữ        | Giải thích                                              | Ví von                               |
| ---------------- | ------------------------------------------------------- | ------------------------------------ |
| **Precision**    | Trong số dự đoán "bệnh", bao nhiêu % đúng?              | "Khi nó nói bệnh, nó có đúng không?" |
| **Recall**       | Trong số thực sự bệnh, phát hiện được bao nhiêu %?      | "Nó có bỏ sót bệnh nào không?"       |
| **mAP**          | Chỉ số tổng hợp chung cho model                         | "Điểm tổng kết"                      |
| **IoU**          | Hộp dự đoán trùng bao nhiêu % với hộp thật?             | "Vẽ ô có chính xác không?"           |
| **Overfitting**  | Model "học thuộc lòng" — giỏi trên train, kém trên test | Học vẹt, không hiểu bài              |
| **Underfitting** | Model chưa học đủ — kém trên cả train lẫn test          | Chưa học xong                        |

### Dữ liệu

| Thuật ngữ           | Giải thích                                                   |
| ------------------- | ------------------------------------------------------------ |
| **Train set**       | Dữ liệu để model học (70-80%)                                |
| **Validation set**  | Dữ liệu kiểm tra trong lúc học (10-15%) — giống thi thử      |
| **Test set**        | Dữ liệu kiểm tra cuối cùng (10-15%) — giống thi thật         |
| **Annotation**      | Nhãn đã gán cho ảnh (vị trí box + tên bệnh)                  |
| **Imbalanced data** | Dữ liệu mất cân bằng (lớp A có 1000 ảnh, lớp B chỉ có 10)    |
| **Augmentation**    | Tăng cường dữ liệu bằng biến đổi ảnh (xoay, lật, đổi màu...) |

---

## 8. Transfer Learning — "Chuyển giao kiến thức"

Đây là kỹ thuật **quan trọng nhất** trong dự án này.

### Ý tưởng:

Thay vì dạy model từ con số 0 (rất lâu, cần rất nhiều dữ liệu), ta:

1. Lấy model đã được train trên **dataset khổng lồ** (COCO: 330K ảnh, 80 loại đối tượng)
2. Model này đã biết nhận diện: mắt, tai, chân, cạnh, góc, texture...
3. Ta chỉ cần **tinh chỉnh** (fine-tune) để nó nhận ra bệnh lá sầu riêng

```
Model COCO (biết 80 thứ)
        │
        ▼ Transfer Learning
Model Sầu Riêng (biết bệnh lá)
```

### Lợi ích:

| Không Transfer Learning   | Có Transfer Learning |
| ------------------------- | -------------------- |
| Cần 100.000+ ảnh          | Cần vài nghìn ảnh    |
| Train vài ngày - vài tuần | Train vài giờ        |
| Dễ overfitting            | Ổn định hơn          |

---

## 9. Tóm tắt flow dự án

```
Bạn có: Ảnh lá sầu riêng + nhãn bệnh
              │
              ▼
pull_data.py: Tải ảnh + nhãn từ internet (Roboflow)
              │
              ▼
plot_class_distribution.py: Xem phân bố → phát hiện mất cân bằng
              │
              ▼
refactor_data.py: Gộp 15 lớp → 6 lớp (bỏ lớp ít mẫu)
              │
              ▼
downsample_large_classes.py: Giảm bớt ảnh lớp quá nhiều
              │
              ▼
model_pretrain.py: Dạy YOLO nhận diện bệnh
              │
              ▼
Kết quả: File best.pt = "bộ não" đã học xong
         → Đưa ảnh mới vào → model nói "Bệnh cháy lá, ở góc trái"
```

> **Đọc tiếp:** [01_pull_data.md](./01_pull_data.md)
