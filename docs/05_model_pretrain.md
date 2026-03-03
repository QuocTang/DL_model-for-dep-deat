# 🧠 File 5: `model_pretrain.py` — Huấn luyện YOLOv11x

> **Mục tiêu:** Hiểu cách huấn luyện model Deep Learning, ý nghĩa từng hyperparameter, và cách đọc kết quả training.  
> **Đây là file QUAN TRỌNG NHẤT trong dự án.**

---

## 📋 Code gốc (có chú thích từng dòng)

```python
import torch                   # Framework Deep Learning (PyTorch)
from ultralytics import YOLO   # Thư viện chuyên dụng cho YOLO

def main():
    # ─── BƯỚC 1: KIỂM TRA GPU ───
    # Kiểm tra xem máy có GPU NVIDIA + CUDA không
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    # → True = có GPU, False = không có (train trên CPU rất chậm)

    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    # → Số lượng GPU (thường = 1)

    print(f"CUDA version: {torch.version.cuda}")
    # → VD: "12.1" — phiên bản CUDA driver

    # ─── BƯỚC 2: LOAD MODEL PRETRAINED ───
    model = YOLO("yolo11x.pt")
    # Tải model YOLOv11x đã được train sẵn trên COCO dataset
    # COCO = 330K ảnh, 80 loại đối tượng (người, xe, chó, mèo...)
    # File "yolo11x.pt" chứa "bộ não" đã biết nhận diện 80 thứ
    # Ta sẽ "dạy thêm" nó nhận diện bệnh lá sầu riêng

    # ─── BƯỚC 3: HUẤN LUYỆN ───
    results = model.train(
        data=r"C:\...\durian-1\data.yaml",  # File cấu hình dataset
        batch=8,              # Số ảnh xử lý cùng lúc
        epochs=100,           # Số vòng lặp training
        device=0,             # Dùng GPU thứ 0 (đầu tiên)
        imgsz=640,            # Resize ảnh về 640×640 pixel
        optimizer="AdamW",    # Thuật toán tối ưu
        lr0=0.005,            # Learning rate ban đầu
        lrf=0.10,             # Learning rate cuối (= lr0 × lrf)
        weight_decay=0.0005,  # Chống overfitting
        warmup_epochs=3,      # 3 epoch đầu tăng dần LR
        cos_lr=True,          # LR giảm theo đường cong cosine
        patience=50,          # Dừng sớm nếu 50 epoch không cải thiện
        seed=42,              # Seed cố định để tái tạo kết quả
        pretrained=True,      # Dùng trọng số pretrained
        save=True,            # Lưu checkpoint sau mỗi epoch
        val=True,             # Chạy validation sau mỗi epoch
        plots=True,           # Tạo biểu đồ kết quả
        name="durian-1-yolo11x"  # Tên thí nghiệm
    )

    # ─── BƯỚC 4: VALIDATION RIÊNG ───
    metrics = model.val(
        data=r"C:\...\durian-1\data.yaml",
        imgsz=640,
        batch=8,
        save_json=True        # Xuất kết quả dạng JSON
    )

    print(metrics)

# ─── ENTRY POINT (điểm bắt đầu) ───
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Cần thiết trên Windows
    multiprocessing.set_start_method('spawn', force=True)
    main()
```

---

## 🧩 Giải thích chi tiết từng khái niệm

### 1. PyTorch là gì?

```
PyTorch = "Bộ dụng cụ" để xây dựng model Deep Learning

Giống như:
  - Bạn muốn xây nhà → cần gạch, xi măng, thợ xây
  - Bạn muốn xây model DL → cần PyTorch (gạch + xi măng + dụng cụ)

PyTorch do Facebook (Meta) phát triển.
Google có TensorFlow — đối thủ cạnh tranh.
```

### 2. Ultralyticsla gì?

```
Ultralytics = Công ty phát triển YOLO
            = Thư viện Python giúp dùng YOLO dễ dàng

Nếu PyTorch là "gạch + xi măng":
  → Ultralytics là "nhà đã xây sẵn, bạn chỉ cần trang trí"

Chỉ với 3 dòng code, bạn đã có thể train YOLO:
  model = YOLO("yolo11x.pt")
  model.train(data="data.yaml", epochs=100)
  model.val()
```

### 3. `YOLO("yolo11x.pt")` — Transfer Learning

```
                    COCO Dataset (330K ảnh)
                    ┌──────────────────┐
                    │ 🚗 🐕 🐈 👤 🚲│
                    │ Model đã biết    │
                    │ nhận diện 80 thứ │
                    └────────┬─────────┘
                             │
                    Download yolo11x.pt
                             │
                             ▼
                    ┌──────────────────┐
                    │ "Bộ não" COCO    │
                    │ Biết: cạnh, góc, │
                    │ texture, hình    │
                    │ dạng, màu sắc... │
                    └────────┬─────────┘
                             │
                    Fine-tune trên dữ liệu sầu riêng
                             │
                             ▼
                    ┌──────────────────┐
                    │ "Bộ não" Sầu Riêng│
                    │ Biết: bệnh cháy  │
                    │ lá, thán thư,    │
                    │ sâu ăn, ...      │
                    └──────────────────┘
```

### 4. Hyperparameters — Từng tham số giải thích cụ thể

#### `batch=8` — Kích thước batch

```
Batch = Số ảnh model xử lý CÙNG LÚC trước khi cập nhật "kiến thức"

Ví von: Giáo viên chấm bài
  batch=1:  Chấm 1 bài → sửa cách dạy → Chấm 1 bài tiếp... (quá chậm)
  batch=8:  Chấm 8 bài → tổng hợp lỗi → sửa cách dạy (cân bằng)
  batch=64: Chấm 64 bài → tổng hợp → sửa (nhanh nhưng "sửa cào bằng")

  Batch lớn hơn:
    + Train nhanh hơn
    - Cần nhiều RAM GPU (VRAM)
    - Có thể "bình quân hóa" quá mức

  batch=8 vì YOLOv11x rất nặng → GPU không đủ VRAM cho batch lớn hơn
```

#### `epochs=100` — Số vòng lặp

```
Epoch = 1 lần model xem qua TOÀN BỘ dataset

Ví von: Đọc sách giáo khoa
  epoch 1:   Đọc lần 1 → hiểu sơ sơ
  epoch 10:  Đọc lần 10 → hiểu khá rõ
  epoch 50:  Đọc lần 50 → hiểu sâu
  epoch 100: Đọc lần 100 → thành chuyên gia (hoặc... học thuộc lòng!)

  Quá ít epoch → Underfitting (chưa học xong)
  Quá nhiều epoch → Overfitting (học thuộc lòng, không hiểu bài)
  → Cần patience (early stopping) để auto dừng đúng lúc
```

#### `imgsz=640` — Kích thước ảnh

```
Tất cả ảnh được RESIZE về 640×640 pixel trước khi đưa vào model

Tại sao?
  - Neural network cần input kích cỡ cố định
  - Ảnh gốc có nhiều kích cỡ khác nhau

  imgsz nhỏ (320):  Nhanh, ít VRAM, nhưng MẤT CHI TIẾT
  imgsz lớn (1280): Chậm, nhiều VRAM, nhưng GIỮ CHI TIẾT
  imgsz=640:        Cân bằng tốt cho đa số bài toán
```

#### `optimizer="AdamW"` — Thuật toán tối ưu

```
Optimizer = "Người hướng dẫn" chỉ model cách cập nhật trọng số

Ví von: Bạn đứng trên đỉnh núi, cần xuống thung lũng (loss thấp nhất)

  SGD (Stochastic Gradient Descent):
    "Cứ đi theo hướng dốc nhất" → đơn giản nhưng hay bị kẹt

  Adam:
    "Đi thông minh hơn SGD — tự điều chỉnh bước nhảy" → phổ biến nhất

  AdamW ← dự án này dùng:
    "Adam + Weight Decay đúng cách" → tốt nhất cho fine-tuning
    Weight Decay = phạt trọng số lớn → chống overfitting
```

#### `lr0=0.005` — Learning Rate ban đầu

```
Learning Rate (LR) = "Kích cỡ bước nhảy" mỗi lần cập nhật trọng số

  ┌──────────────────────────────────────────┐
  │                                          │
  │    LR quá lớn (0.1):                     │
  │    ╲   ╱╲   ╱╲   ← Nhảy qua nhảy lại   │
  │     ╲_╱  ╲_╱      Không bao giờ hội tụ!  │
  │                                          │
  │    LR vừa phải (0.005):  ← dự án dùng   │
  │    ╲                                     │
  │     ╲__                                  │
  │        ╲___                              │
  │            ╲    ← Giảm dần, hội tụ tốt   │
  │                                          │
  │    LR quá nhỏ (0.00001):                 │
  │    ──────────── ← Gần như đứng yên       │
  │     Quá chậm, train mãi không xong!      │
  └──────────────────────────────────────────┘
```

#### `lrf=0.10` — Learning Rate cuối cùng

```
LR cuối = LR đầu × lrf = 0.005 × 0.10 = 0.0005

Ý tưởng: Đầu tiên bước TO để tiến nhanh, cuối cùng bước NHỎ để chính xác

Ví von: Đậu xe
  - Lúc đầu → lái nhanh tới gần chỗ đậu (LR lớn)
  - Gần tới → lái chậm, căn chỉnh cẩn thận (LR nhỏ)
```

#### `warmup_epochs=3` — Giai đoạn khởi động

```
3 epoch đầu tiên, LR TĂNG DẦN từ gần 0 lên lr0

  LR
  │        lr0 = 0.005
  │       ╱─────────────────
  │      ╱
  │     ╱   Warmup
  │    ╱    (tăng dần)
  │   ╱
  │  ╱
  │ ╱
  │╱____________________________
  0    1    2    3    4    5   → Epoch

Tại sao?
  - Model mới load pretrained → rất "nhạy cảm"
  - Nếu dùng LR lớn ngay → có thể "phá hỏng" kiến thức pretrained
  - Warmup = cho model "làm quen" từ từ
```

#### `cos_lr=True` — Cosine Learning Rate Schedule

```
LR giảm theo đường cong cosine (lượn sóng) thay vì giảm đột ngột

  LR
  │ ╲     Cosine (mượt mà) ← dùng cái này
  │  ╲
  │   ╲╲
  │     ╲╲╲
  │        ╲╲╲╲╲__________
  └──────────────────────── Epoch

  So với Step Decay (giảm đột ngột):
  │ ────┐
  │     │────┐
  │          │────┐
  │               │────
  └──────────────────────── Epoch

  Cosine tốt hơn vì model được "chuyển tiếp mượt mà" hơn.
```

#### `patience=50` — Early Stopping

```
"Nếu 50 epoch liên tiếp validation mAP KHÔNG CẢI THIỆN → DỪNG TRAINING"

  mAP
  │          ╱──────── ← Đạt đỉnh, không tăng nữa
  │        ╱
  │      ╱
  │    ╱
  │  ╱
  │╱
  └─────────────────── Epoch
       ↑              ↑
   Epoch 60      Epoch 110
   (đạt đỉnh)   (50 epoch không cải thiện → DỪNG!)

  Tại sao cần?
  - Tiết kiệm thời gian (không train vô ích)
  - Tránh overfitting (train quá lâu → model "học thuộc lòng")

  patience=50 khá cao → cho phép "chờ đợi" lâu trước khi dừng
```

#### `weight_decay=0.0005` — Regularization

```
Weight Decay = phạt model nếu trọng số quá lớn

Ví von: Viết luận văn
  - Không weight decay: Viết dài, rườm rà, nhiều chi tiết thừa
  - Có weight decay: Viết ngắn gọn, súc tích, chỉ giữ ý chính

  → Tránh overfitting (model quá phức tạp, "nhớ" cả noise trong data)

  0.0005 = mức phạt nhẹ (đủ để regularize nhưng không ảnh hưởng learning)
```

#### `seed=42` — Random Seed

```
seed = số "hạt giống" cho random

random.seed(42)
random.random()  # → Luôn ra 0.6394... (mỗi lần chạy đều giống nhau)

Tại sao cần?
  - Đảm bảo kết quả CÓ THỂ TÁI TẠO (reproducible)
  - Người khác chạy cùng code + cùng seed → cùng kết quả

Tại sao 42?
  - Đùa vui: Trong "The Hitchhiker's Guide to the Galaxy",
    42 là "đáp án cho mọi thứ trong vũ trụ" 😄
  - Thực tế: bất kỳ số nào cũng được, 42 là convention phổ biến
```

### 5. Validation — Kiểm tra model

```python
metrics = model.val(
    data=r"...\data.yaml",
    imgsz=640,
    batch=8,
    save_json=True    # Xuất kết quả chi tiết dạng JSON
)
```

```
Sau khi train xong, chạy val() để đánh giá model trên validation set:

  metrics sẽ chứa:
  ├── mAP@50:     Chỉ số tổng hợp (IoU ≥ 50%)
  ├── mAP@50-95:  Chỉ số tổng hợp (trung bình IoU 50-95%)
  ├── Precision:  Dự đoán bệnh → bao nhiêu % đúng?
  ├── Recall:     Thực sự bệnh → phát hiện bao nhiêu %?
  └── Per-class:  Metrics cho từng lớp riêng
```

### 6. `if __name__ == '__main__':` — Entry Point

```python
if __name__ == '__main__':
    # Code ở đây CHỈ chạy khi bạn chạy FILE NÀY TRỰC TIẾP
    # KHÔNG chạy khi file này được IMPORT từ file khác

    import multiprocessing
    multiprocessing.freeze_support()   # Windows cần dòng này
    multiprocessing.set_start_method('spawn', force=True)
    # 'spawn' = tạo process mới hoàn toàn (an toàn trên Windows)
    main()
```

```
Tại sao cần multiprocessing?
  - YOLO dùng nhiều worker processes để LOAD DỮ LIỆU song song
  - Worker 1 load ảnh batch 1, Worker 2 load ảnh batch 2...
  - Trong lúc GPU xử lý batch 1, CPU đã sẵn sàng batch 2
  - → Không lãng phí thời gian chờ load ảnh

  ┌─────────────────────────────────────────────┐
  │ Worker 1: [Load batch 1][Load batch 3]...   │
  │ Worker 2:     [Load batch 2][Load batch 4]  │
  │ GPU:      [Process 1][Process 2][Process 3] │
  │           ← Không có khoảng trống "chờ" →   │
  └─────────────────────────────────────────────┘
```

---

## 📈 Kết quả dự án — Cách đọc hiểu

### File `results.csv` — Log training

```
epoch | train/box_loss | train/cls_loss | mAP50  | mAP50-95
──────┼────────────────┼────────────────┼────────┼──────────
  1   |     2.224      |     3.441      | 0.004  |  0.001
  50  |     1.266      |     1.543      | 0.181  |  0.107
  100 |     0.988      |     0.908      | 0.306  |  0.198

Đọc: Box loss và cls loss GIẢM dần ✅ → model đang học
      mAP50 TĂNG từ 0.4% → 30.6% ✅ → model cải thiện
      Nhưng 30.6% VẪN THẤP ❌ → cần cải thiện thêm
```

### 3 loại Loss trong YOLO

```
┌─────────────────────────────────────────────────────────┐
│ box_loss (Loss vị trí):                                 │
│   "Hình chữ nhật tôi vẽ có đúng vị trí không?"        │
│   Đo sai lệch giữa predicted box và ground truth box   │
│                                                         │
│ cls_loss (Loss phân loại):                              │
│   "Tôi gọi tên bệnh có đúng không?"                   │
│   Đo sai lệch giữa predicted class và actual class     │
│                                                         │
│ dfl_loss (Distribution Focal Loss):                     │
│   Loss nâng cao cho vị trí box (chi tiết hơn box_loss) │
│   Giúp model dự đoán biên box CHÍNH XÁC hơn           │
└─────────────────────────────────────────────────────────┘

Cả 3 loss đều PHẢI GIẢM trong quá trình training.
Nếu loss TĂNG → model đang có vấn đề!
```

### Metrics đánh giá

```
Precision = 68.6%
  "Khi model nói 'Đây là bệnh cháy lá', nó đúng 68.6% lần"
  → Khá ổn, nhưng vẫn sai 31.4%

Recall = 28.1%
  "Trong 100 vùng thực sự bị bệnh, model chỉ phát hiện 28 vùng"
  → RẤT KÉM! Bỏ sót 72% bệnh!

mAP@50 = 30.6%
  "Điểm tổng kết" — kết hợp Precision + Recall
  → THẤP (cần > 50% mới tạm chấp nhận)

mAP@50-95 = 19.8%
  "Điểm tổng kết khắt khe hơn"
  → RẤT THẤP
```

---

## 💡 Tóm tắt dòng chảy training

```
EPOCH 1:
  Model: "Tôi không biết gì về bệnh lá sầu riêng"
  Loss: RẤT CAO (sai rất nhiều)
  mAP: GẦN 0% (gần như đoán mò)

EPOCH 50:
  Model: "Tôi bắt đầu nhận ra một số bệnh"
  Loss: GIẢM (học dần)
  mAP: ~18% (có tiến bộ nhưng vẫn kém)

EPOCH 100:
  Model: "Tôi biết sâu ăn khá tốt, nhưng nhiều bệnh khác vẫn lẫn"
  Loss: THẤP (train tốt)
  mAP: ~30% (cải thiện nhưng chưa đủ)

  → Cần: Thêm dữ liệu, cân bằng tốt hơn, hoặc điều chỉnh hyperparams
```

---

> **Xong!** Bạn đã đọc hết toàn bộ tài liệu. Quay lại [README](./README.md) để ôn tập.
