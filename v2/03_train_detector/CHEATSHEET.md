# 📋 CHEAT SHEET — Bước 3 (in 2-3 trang mang theo)

## 🆘 CÂU CỨU MẠNG (học thuộc, bí thì đọc)

> **"Bước 3 train YOLO11n (model detection nano, đã pretrained trên COCO) trên data lá refactor 6 class. Smoke test 8 epochs, imgsz=320, batch=16. Kết quả: mAP50 ≈ 30%, đủ để Bước 4 lấy feature."**

---

## 📖 Câu chuyện 1 phút (analogy)

```
yolo11n.pt (pretrained)        =  Học sinh đã tốt nghiệp cấp 3
                                  (biết đếm, viết, nhận diện vật thể chung)

Fine-tune trên data lá         =  Dạy thêm chuyên môn về bệnh lá sầu riêng
                                  (8 vòng học = 8 epochs)

Output: best.pt                =  Học sinh tốt nghiệp chuyên ngành
                                  (giờ chuyên nhận diện bệnh lá)
```

→ **YOLO** = "**You Only Look Once**" — model detection 1-stage, vẽ box + class trong 1 lần forward.

---

## 🔧 File `train_yolo.py` làm 5 việc

1. **Check GPU**: in ra "có CUDA không, mấy card".
2. **Load YOLO pretrained** (`yolo11n.pt` — phiên bản "nano", ~2.6M tham số).
3. **Train 8 epochs** trên data lá refactor (1 epoch = đi qua toàn bộ ~2400 ảnh).
4. **Validate** trên test set sau khi train xong → save `best.pt`.
5. **In metrics**: mAP50, mAP50-95, Precision, Recall.

→ **Output chính**: `best.pt` (~5MB), input cho Bước 4.

---

## 🔢 3 số PHẢI nhớ thuộc

| Số | Câu nói |
|---|---|
| **8 epochs** | "Em train 8 vòng — smoke test, production cần 50-100." |
| **imgsz=320** | "Resize 320×320 cho nhanh trên GTX 1650 (4GB VRAM)." |
| **mAP50 ≈ 0.295** | "Kết quả cuối: mAP50 ~ 30%, đủ để qua Bước 4." |

---

## ❓ 8 câu Q&A có sẵn

**"YOLO là gì?"**
> "You Only Look Once" — model detection 1-stage, vẽ box + class trong 1 lần forward. Nhanh nhất trong các detector.

**"Tại sao dùng YOLO11n mà không phải s/m/l?"**
> "n" = nano, ~2.6M tham số, nhanh nhất, hợp với GPU yếu (GTX 1650 4GB). Production có thể đổi sang `s` hoặc `m` cho accuracy cao hơn.

**"Epoch là gì?"**
> 1 epoch = model đi qua **toàn bộ** data train **1 lần**. 8 epochs = 8 lần ôn bài.

**"Batch là gì?"**
> Số ảnh model xem **cùng lúc** trước khi cập nhật weight. `batch=16` = mỗi step xem 16 ảnh.

**"Learning rate là gì?"**
> **Bước nhảy** mỗi lần update. Cao → học nhanh, dễ vọt qua optimum. Thấp → chậm, ổn định.

**"Pretrained có ý nghĩa gì?"**
> Bắt đầu từ weights đã train trên 80 class COCO, **không random**. Tiết kiệm data + thời gian. Như học sinh đã có nền.

**"Fine-tune khác training from scratch?"**
> Fine-tune = train tiếp model có sẵn. From scratch = train từ đầu (weight random). Fine-tune cần ít data hơn rất nhiều.

**"mAP50 là gì?"**
> mean Average Precision với IoU ≥ 0.5. Trung bình precision của 6 class khi box dự đoán trùng box thật ≥ 50%. **Số chính** đánh giá detector.

---

## 🎛 Hyperparameters — chia 3 nhóm

### **Nhóm A — Quy mô training** (lớn nhất quyết định kết quả)

| Param | Code | Analogy / Ý nghĩa |
|---|---|---|
| `epochs` | **8** | Đi qua toàn bộ data **8 lần** = ôn 8 vòng |
| `imgsz` | **320** | Resize ảnh **320×320** trước khi học |
| `batch` | **16** | Xem **16 ảnh cùng lúc** trước khi update weight |
| `fraction` | **1.0** | Dùng **100%** data train |

### **Nhóm B — Optimizer** (cách model "học")

| Param | Code | Analogy |
|---|---|---|
| `optimizer` | **"AdamW"** | Thuật toán update weight, phổ biến nhất hiện nay |
| `lr0` | **0.005** | **Learning rate ban đầu** = bước nhảy lúc khởi đầu |
| `lrf` | **0.10** | **lr cuối** = lr0 × 0.10 = 0.0005, học chậm dần |
| `cos_lr` | **True** | Giảm lr theo đường cong **cosine** (mượt, không đột ngột) |
| `weight_decay` | **0.0005** | **Phạt nhẹ** weight quá lớn → chống overfit |
| `warmup_epochs` | **1** | Epoch đầu khởi động chậm rồi mới tăng tốc |

### **Nhóm C — Kiểm soát training**

| Param | Code | Ý nghĩa |
|---|---|---|
| `pretrained` | **True** | Bắt đầu từ weights đã train trên COCO, không random |
| `seed` | **42** | Hạt giống reproducibility, chạy lại ra y hệt |
| `cache` | **"ram"** | Load hết ảnh vào RAM → nhanh hơn 2-3× |
| `patience` | **8** | Dừng sớm nếu 8 epoch không cải thiện |
| `save_period` | **-1** | Chỉ save `best.pt` + `last.pt`, không save mỗi epoch |
| `val` | **True** | Tự validate sau mỗi epoch |
| `plots` | **True** | Vẽ confusion matrix, P-R curve... |

### Câu giải thích cho thầy hỏi sâu

**`epochs=8`**
> Smoke test cho nhanh. Production cần 50-100 để model "học chín". Mỗi epoch ~45 giây trên GTX 1650.

**`imgsz=320`**
> Half size so với chuẩn 640 → compute giảm 4×. Hi sinh chi tiết nhỏ để chạy nhanh trên GPU yếu.

**`batch=16`**
> Vừa đủ cho 4GB VRAM của GTX 1650. Batch lớn hơn → train ổn định hơn nhưng cần GPU mạnh.

**`lr0=0.005 + lrf=0.10 + cos_lr=True`**
> Combo learning rate: bắt đầu 0.005, giảm cosine xuống 0.0005 (= lr0×lrf). Khởi đầu nhanh, cuối tinh chỉnh.

**`AdamW`**
> Phiên bản cải tiến của Adam, decoupled weight decay → tốt hơn cho transformer + modern model. Default tốt cho hầu hết bài.

**`weight_decay=0.0005`**
> Regularization L2 nhẹ, phạt weight quá lớn để chống overfit.

**`pretrained=True`**
> Bắt buộc cho fine-tuning. False = train from scratch, cần >10× data.

---

## 📊 Cách đọc metrics

### Sau train xong, sẽ thấy:

```
Test mAP50    : 0.295    ← SỐ CHÍNH
Test mAP50-95 : 0.163
Test P (Precision) : 0.324
Test R (Recall)    : 0.428
```

### Mỗi số nghĩa là gì?

| Metric | Hỏi gì? | Cao = ? |
|---|---|---|
| **Precision (P)** | "Khi model vẽ hộp, đúng %?" | Đoán chắc, ít bắt nhầm |
| **Recall (R)** | "Trong các hộp thật, bắt được %?" | Bắt nhiều, ít bỏ sót |
| **mAP50** | mean AP với IoU ≥ 0.5 (box trùng ≥ 50%) | **SỐ CHÍNH** đánh giá detector |
| **mAP50-95** | mAP trung bình từ IoU 0.5 → 0.95 | KHẮC NGHIỆT hơn (đòi box trùng cao) |

### Trade-off P vs R (như XGBoost ở Bước 5)

```
P cao, R thấp:  Đoán siêu chắc, bỏ sót nhiều  (overcautious)
P thấp, R cao:  Đoán bừa, bắt nhiều           (overeager)
P+R cân bằng:                                 ✅
mAP combine cả 2 → là số chính
```

### Tiến triển 8 epoch của smoke-test này

```
       Ep1   Ep2   Ep3   Ep4   Ep5   Ep6   Ep7   Ep8
P      5%    64%   33%   22%   20%   28%   32%   32%
R      9%    14%   23%   38%   37%   40%   44%   43%
mAP50  2%    10%   16%   21%   21%   27%   30%   30%
```

→ Trend **tăng dần đều**, chưa ổn định. Tăng epochs sẽ tiếp tục cải thiện.

---

## ⚠️ ĐỪNG NÓI NHẦM

- ❌ "YOLO classify ảnh" → ✅ YOLO **detect** (vẽ box + class), không phải classify cả ảnh.
- ❌ "Em train từ đầu" → ✅ Em **fine-tune** từ pretrained `yolo11n.pt`.
- ❌ "8 epochs là đủ" → ✅ 8 là **smoke test**, production cần 50-100.
- ❌ "mAP50 30% là model dở" → ✅ Bài detection 6 class với 8 epochs trên GPU yếu, **30% là OK** cho smoke test.
- ❌ "Batch là số class" → ✅ Batch là số **ảnh xem cùng lúc** (= 16). Số class là 6, riêng biệt.
- ❌ "Learning rate là số epoch" → ✅ lr là **bước nhảy** mỗi lần update, không liên quan epoch.

---

## 📈 Bias-aware: YOLO yếu ở class hiếm

Sau train, theo class:

| Class | Recall | Nhận xét |
|---|---:|---|
| `benh-chay-la` (778 ảnh) | ~50% | OK |
| `sau-an` (639 ảnh) | ~40% | OK |
| `benh-than-thu` | ~30% | Trung bình |
| `benh-dom-mat-cua` | **3%** | ❌ Bỏ sót |
| `la-khoe-binh-thuong` | **0%** | ❌ Bỏ sót hoàn toàn |
| `others` | ~20% | Yếu |

→ Đây là lý do **cần Bước 4 + 5**: XGBoost classify cả ảnh sẽ cứu được class hiếm (la-khoe lên 80% recall).

---

## 📦 Pipeline tổng (slide cuối)

```
Bước 1 (refactor 15→6 class)
    ↓
Bước 2 (EDA: imbalance 18×)
    ↓
Bước 3 (Train YOLO: mAP50 ≈ 30%)   ← MÌNH ĐANG ĐÂY
    ↓                  ⚠️ class hiếm còn yếu
Bước 4 (Bóp não YOLO → vector 512)
    ↓
Bước 5 (XGBoost: Acc 65%, F1 63%)
    ↓                  ⭐ giải cứu la-khoe (0% → 80%)
Bước 6 (Inference end-to-end)
```

---

## 💡 Mini-định nghĩa Bước 3

| Từ | 1 câu |
|---|---|
| **YOLO** | "You Only Look Once" — model detection 1-stage, nhanh nhất. |
| **Detection** | Vẽ box + gán class cho từng vật trong ảnh. |
| **Classification** | Gán 1 nhãn cho cả ảnh (Bước 5 làm). |
| **Pretrained** | Weight đã train sẵn trên dataset khác (COCO). |
| **Fine-tune** | Train tiếp model pretrained trên data của mình. |
| **Epoch** | 1 lần đi qua toàn bộ data train. |
| **Batch** | Số ảnh xem cùng lúc trước khi update weight. |
| **Learning rate** | Bước nhảy mỗi lần update weight. |
| **Optimizer** | Thuật toán điều khiển cách update weight (AdamW, SGD). |
| **Cosine lr** | Schedule giảm lr theo đường cong cosine. |
| **Warmup** | Epoch đầu khởi động chậm để training ổn định. |
| **Weight decay** | L2 regularization, phạt weight lớn. |
| **mAP50** | Mean Average Precision với IoU ≥ 0.5. |
| **IoU** | Intersection over Union — box dự đoán trùng box thật bao nhiêu %. |
| **Precision** | "Khi nói 'có bệnh', đúng %?" |
| **Recall** | "Trong các bệnh thật, bắt được %?" |
| **Patience (early stopping)** | Dừng sớm nếu N epoch không cải thiện. |
| **best.pt** | Weight tốt nhất (theo valid mAP). |
| **last.pt** | Weight epoch cuối (có thể overfit). |

---

## 🩺 Triệu chứng → Hành động (nếu thầy hỏi tune)

```
┌─ mAP thấp (< 50%) ──────┬─ Hành động ──────────────────────┐
│                         │ ↑ epochs (8 → 50-100)            │
│                         │ ↑ imgsz (320 → 640)              │
│                         │ Đổi model n → s → m              │
├─────────────────────────┼──────────────────────────────────┤
│ Class hiếm 0% recall    │ Tăng data class đó (augmentation)│
│                         │ Class weight trong loss          │
│                         │ → Hoặc dùng Bước 4+5 cứu (XGB)   │
├─────────────────────────┼──────────────────────────────────┤
│ Train mAP cao,          │ ↓ epochs                         │
│ Test mAP thấp (overfit) │ ↑ weight_decay                   │
│                         │ Augmentation                     │
└─────────────────────────┴──────────────────────────────────┘
```

---

> **Bình tĩnh — bạn đã nắm câu chuyện rồi 💪**
> Hỏi gì cũng quay về CÂU CỨU MẠNG ở đầu trang.
