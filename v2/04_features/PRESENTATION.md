# 04 — Bản tóm tắt khi thuyết trình (~5 phút nói)

> File này chỉ chứa **ý chính để nói** + **số liệu cần nhớ** + **câu hỏi dự kiến**.
> Cần giải thích sâu? Đọc `NOTES.md` cùng thư mục.

---

## 🎯 Elevator pitch (30 giây — học thuộc)

> "Bước 4 là **cây cầu** giữa YOLO (Bước 3) và XGBoost (Bước 5). YOLO đã train xong, mình **mượn 3 layer giữa** của nó để biến mỗi ảnh thành **vector 512 số**, rồi lưu thành CSV để Bước 5 train classifier."

---

## 🗣 5 ý chính khi trình bày

### 1. Tại sao cần Bước 4? (45s)
- YOLO làm **detection** (vẽ hộp + class cho từng vật) — không phải classification.
- Mình muốn **1 nhãn / ảnh** bằng **XGBoost** (nhanh, chạy CPU, dễ giải thích).
- XGBoost chỉ ăn **vector số**, không hiểu ảnh thô.
- → Cần "phiên dịch": **ảnh → vector**.

### 2. Ý tưởng "mượn não YOLO" (45s)
- YOLO đã train trên data của mình → biết nhìn lá bệnh.
- **Backbone** (layer 0-9) = phần "nhìn" tổng quát → dùng được cho task khác.
- **Head** (layer 10+) = phần ra hộp detection → quá đặc thù, bỏ.
- Lấy output **layer 4, 6, 9** — vùng học texture + hình nhỏ (vừa đủ phân biệt bệnh, vừa đủ tổng quát).

### 3. Pipeline 1 ảnh đi qua (1 phút)
```
Ảnh gốc → letterbox 320×320 → YOLO backbone forward
                                    ↓
        Hook chộp 3 feature map: [64×40×40] [128×20×20] [320×10×10]
                                    ↓
        AdaptiveAvgPool 1×1 → 64 số + 128 số + 320 số
                                    ↓
                            Concat = vector 512
                                    ↓
                           Lưu 1 dòng vào CSV
```

### 4. Kết quả (30s)
- 3 CSV: `train (2359 ảnh)`, `valid (227)`, `test (112)`.
- Mỗi dòng = **515 cột** = 512 feature + label + class_name + image_name.
- 6 class: `sau-an`, `benh-chay-la`, `benh-than-thu`, `benh-dom-mat-cua`, `la-khoe-binh-thuong`, `others`.

### 5. Phát hiện thú vị: Imbalance giảm mạnh (45s)
- EDA (theo annotation): tỉ lệ class lớn / class nhỏ = **18×**.
- Sau Bước 4 (theo ảnh): chỉ còn **3.9×**.
- Lý do: 1 ảnh có nhiều box cùng class → đếm theo ảnh thì gộp.
- → **Bước 5 sẽ dễ hơn YOLO ở Bước 3**.

---

## 🔢 Số liệu cần nhớ

| Số | Ý nghĩa |
|---|---|
| **6** | số class sau khi refactor (Bước 1) |
| **512** | chiều của vector feature (= 64 + 128 + 320) |
| **3** | số layer chộp (4, 6, 9) |
| **320** | imgsz — phải khớp với train ở Bước 3 |
| **18× → 3.9×** | imbalance giảm |
| **2359 / 227 / 112** | số ảnh train / valid / test |
| **~15 phút** | thời gian chạy hết 2700 ảnh trên GTX 1650 |

---

## 🧠 4 thuật ngữ cốt lõi (giải thích bằng 1 câu)

| Thuật ngữ | Câu giải thích |
|---|---|
| **Hook** | "Camera đặt giữa pipeline để chộp output layer mà không phá code gốc." |
| **Feature map** | "Stack các bản đồ nhiệt — mỗi tấm đánh dấu vùng kích hoạt 1 pattern." |
| **Letterbox** | "Resize giữ tỉ lệ + pad xám 114, không squish ảnh méo." |
| **Dominant class** | "1 ảnh có nhiều box → lấy class xuất hiện nhiều nhất làm nhãn." |

---

## ❓ Q&A dự kiến (đọc trước, đừng hoang mang)

**Q: Tại sao chọn layer 4, 6, 9 mà không phải khác?**
> Layer 0-3 quá thấp (chỉ thấy cạnh, mọi ảnh giống nhau). Layer 10+ quá đặc thù cho detection. 4-9 ở giữa học texture/hình nhỏ — sweet spot. Lấy 3 layer để có đa độ phân giải.

**Q: Tại sao vector 512 chứ không phải 1024 hay 256?**
> Đây là tổng channel của layer 4, 6, 9 trong YOLO11n: 64 + 128 + 320 = 512. Không phải số tự chọn — model quyết định.

**Q: Tại sao avg-pool 1×1 mà không phải max-pool, hay flatten thẳng?**
> Mình không cần vị trí ("vết bệnh ở góc nào") — chỉ cần "ảnh này có pattern không". Flatten thẳng sẽ ra vector quá to (51,200 chiều) → chậm + dễ overfit. Avg ổn định hơn max.

**Q: Tại sao không train CNN classifier head luôn cho gọn?**
> Nặng (cần GPU), chậm, và XGBoost trên deep feature thường ngang ngửa hoặc tốt hơn cho dataset nhỏ vài nghìn ảnh. Combo "feature từ deep + classifier cổ điển" là pattern phổ biến.

**Q: Tại sao imgsz phải đúng 320?**
> Vì Bước 3 train ở 320. Đưa size khác (vd 640) → activation các layer thay đổi → vector "lệch" so với phân bố model đã học → ML sai.

**Q: Sao không dùng `model.predict()` của Ultralytics cho gọn?**
> `predict()` chỉ trả về detection cuối, không cho output layer giữa. Phải dùng PyTorch hook để chộp.

**Q: Có thể dùng cho dataset khác không?**
> Có. Đổi `--model` (best.pt khác) + `--data-yaml` hoặc `--dataset-root`. Script tự auto-detect format YOLO/TensorFlow.

---

## ⚠️ Đừng nói nhầm

- KHÔNG nói "YOLO classify ảnh" → YOLO **detect**, mình tự nắn về classification.
- KHÔNG nói "fine-tune YOLO ở Bước 4" → **không train gì** ở Bước 4, chỉ inference.
- KHÔNG nói "vector 448" → đoán ngây thơ 64+128+256, **thực tế** YOLO11n là 64+128+**320** = 512.
- KHÔNG nói "classification dễ vì ít data" → classification dễ hơn vì **imbalance giảm** (18× → 3.9×) và **1 nhãn/ảnh đơn giản hơn detection**.

---

## 🎬 Kịch bản demo (nếu có máy)

```bash
cd v2
uv run 04_features/extract_deep_features.py --imgsz 320 --max-images 20
```
→ Ra CSV nhanh trong ~30 giây, mở `train_features.csv` cho khán giả thấy 515 cột thực tế.

---

## 📌 Kết nối với toàn pipeline (slide cuối)

```
Bước 1 (Refactor 15→6 class)
    ↓
Bước 2 (EDA: imbalance 18×)
    ↓
Bước 3 (Train YOLO detector — mAP50 = 0.295)
    ↓
Bước 4 (Bóp não YOLO → 512-dim vector)   ← MÌNH ĐANG ĐỨNG ĐÂY
    ↓
Bước 5 (Train XGBoost classifier — sẽ dễ hơn nhờ imbalance giảm)
    ↓
Bước 6 (Inference end-to-end)
```

> Câu chốt: **"Bước 4 là khoảnh khắc dataset của mình thoát khỏi pixel, bước vào không gian số mà mọi ML cổ điển có thể xử lý."**
