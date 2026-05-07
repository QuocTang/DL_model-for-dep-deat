# 📋 CHEAT SHEET — Bước 4 (in 1 trang mang theo)

## 🆘 CÂU CỨU MẠNG (học thuộc, bí thì đọc)

> **"Bước 4 dùng YOLO đã train ở Bước 3 như một bộ trích xuất đặc trưng — biến mỗi ảnh thành vector 512 số. Vector này là input cho XGBoost ở Bước 5 để phân loại."**

---

## 📖 Câu chuyện 1 phút (analogy)

- **Robot A (YOLO)**: thông minh, nhìn ảnh được. Đã train xong ở Bước 3.
- **Robot B (XGBoost)**: đơn giản, chỉ ăn SỐ, không nhìn ảnh.
- **Bước 4 = cái cầu**: bảo Robot A mô tả ảnh thành **512 số**, đưa cho Robot B.

---

## 🔧 File làm 5 việc

1. Mở YOLO `best.pt` đã train.
2. Đặt 3 "máy quay" (hook) vào layer 4, 6, 9 — chỗ giữa não YOLO.
3. Cho từng ảnh đi vào → 3 máy quay tự chộp output.
4. Gom 3 cái chộp → trung bình → ra **512 số** (= 64 + 128 + 320).
5. Ghi vào CSV cùng nhãn (sâu ăn / cháy lá / ...).

→ Output: 3 file CSV `train/valid/test_features.csv`. Sẵn sàng cho Bước 5.

---

## 🔢 3 SỐ phải nhớ

| Số | Câu nói |
|---|---|
| **512** | "Mỗi ảnh thành vector 512 số." |
| **3 layer (4,6,9)** | "Lấy từ 3 chỗ giữa của YOLO." |
| **6 class** | "Đầu ra 6 nhãn bệnh." |

**Phụ:** 64+128+320 = 512 (tổng channel 3 layer).

---

## ❓ 8 câu Q&A có sẵn

**"Tại sao cần Bước 4?"**
> XGBoost không nhìn ảnh được, chỉ ăn số. Cần biến ảnh → vector số.

**"Tại sao 512?"**
> Tổng channel của layer 4, 6, 9 trong YOLO11n: 64+128+320=512.

**"Tại sao layer 4, 6, 9?"**
> Layer 0-3 quá thấp (chỉ thấy cạnh). Layer 10+ chuyên cho detection. 4-9 ở giữa = sweet spot, học texture lá + đốm bệnh.

**"Hook là gì?"**
> Là tính năng PyTorch — "móc" vào giữa model để chộp output layer giữa, không sửa code YOLO gốc.

**"Channel là gì?"**
> Là 1 "ảnh xám nhỏ" mà 1 filter tạo ra. Layer 4 có 64 filter → 64 channel.

**"Tại sao không train CNN classifier luôn?"**
> CNN cần GPU + nhiều data. Mình ~2700 ảnh + chạy CPU → XGBoost trên feature deep ổn hơn.

**"Layer 4, 6, 9 là loại layer gì?"**
> Đều là **Convolutional layer** (có filter, đã train sẵn ở Bước 3). Em chỉ "mượn" output của chúng.

**"Có dùng Fully Connected không?"**
> KHÔNG. Em áp **Average Pooling** sau hook, rồi đưa vector 512 cho **XGBoost** ở Bước 5. XGBoost thay thế vai trò của Fully Connected.

---

## ⚠️ ĐỪNG NÓI NHẦM

- ❌ "YOLO classify ảnh" → ✅ YOLO **detect**, mình tự nắn về classification.
- ❌ "Bước 4 train YOLO" → ✅ Bước 4 **không train gì**, chỉ inference (mượn não).
- ❌ "Vector 448 chiều" → ✅ **512** (64+128+320, không phải 64+128+256).
- ❌ "Em chọn 512" → ✅ Số 512 là **kết quả** của kiến trúc YOLO11n, không phải em chọn.
- ❌ "Layer mình dùng là FC / pooling" → ✅ Layer 4,6,9 là **Convolutional**. Pooling là phép áp **sau** khi hook.
- ❌ "Avg connect / pooling connect" → ✅ Không có thuật ngữ này. Chỉ có **Conv layer / Pooling layer / FC layer**.

---

## 🧱 3 LOẠI LAYER trong CNN (phân biệt 1 lần là xong)

| Loại layer | Có học | Có filter/kernel | Để làm gì |
|---|---|---|---|
| **Convolutional (Conv)** | ✅ | ✅ **CÓ filter** | Phát hiện pattern (cạnh, đốm…) |
| **Pooling (Avg/Max)** | ❌ | ❌ KHÔNG filter | Nén size, lấy trung bình hoặc max |
| **Fully Connected (FC)** | ✅ | ❌ KHÔNG filter | Tổng hợp ra class cuối |

→ **"Filter" CHỈ xuất hiện ở Conv**. Pooling và FC **không có filter**.

**Trong file `extract_deep_features.py` dùng cái nào?**
- Layer 4, 6, 9 (mình hook) = **Conv** (filter đã train ở Bước 3, mình mượn dùng).
- Sau hook = **AdaptiveAvgPool2d(1,1)** = **Average Pooling** (không học, chỉ trung bình).
- **KHÔNG dùng FC** — XGBoost ở Bước 5 thay thế FC.

**So sánh "filter có học" vs "không học":**

```
Conv (CÓ học):                      Avg Pooling (KHÔNG học):
┌──────────────────┐                ┌──────────────────┐
│ Filter 3×3:      │                │ Lấy trung bình   │
│  [w1 w2 w3]      │                │ N ô:             │
│  [w4 w5 w6] ←học │                │  [a b c d]       │
│  [w7 w8 w9]      │                │  → (a+b+c+d)/4   │
│                  │                │                  │
│ w thay đổi qua   │                │ Không có w gì cả │
│ training         │                │ — phép cố định   │
└──────────────────┘                └──────────────────┘
```

⚠️ "Fully Connected" là **TÊN layer**, KHÔNG phải kiểu "connect". Không có thuật ngữ "avg connect" / "pooling connect" trong giáo trình.

---

## 📦 Pipeline tổng (slide cuối)

```
Bước 1 (refactor 15→6 class)
    ↓
Bước 2 (EDA: imbalance 18×)
    ↓
Bước 3 (Train YOLO: mAP50 = 0.295)
    ↓
Bước 4 (Bóp não YOLO → vector 512)   ← MÌNH ĐANG ĐÂY
    ↓
Bước 5 (XGBoost: Test Acc 65%)
    ↓
Bước 6 (Inference end-to-end)
```

---

## 💡 Mini-định nghĩa (nếu bí)

| Từ | 1 câu |
|---|---|
| **PyTorch** | Thư viện Python để xây/chạy mạng neural; YOLO viết bằng PyTorch. |
| **CNN** | Mạng neural cho ảnh, gồm nhiều layer (filter) xếp chồng. |
| **Filter (kernel)** | Ma trận nhỏ trượt qua ảnh để tìm 1 pattern (cạnh, đốm…). |
| **Layer** | 1 hộp chứa nhiều filter (vd layer 4 có 64 filter). |
| **Channel** | 1 ảnh xám = output của 1 filter. |
| **Feature map** | Stack nhiều channel của 1 layer. |
| **Hook** | Cái "máy quay" chộp output layer giữa (PyTorch feature). |
| **Letterbox** | Resize giữ tỉ lệ + pad xám 114, không squish ảnh méo. |
| **Pooling** | Trung bình/max của 1 vùng → giảm size, lấy info tổng quát. |
| **Forward** | Cho 1 ảnh đi qua model từ đầu đến cuối, lấy output. |
| **Inference** | Dùng model đã train để dự đoán (không train thêm). |
| **Backbone** | Phần "nhìn" của YOLO (layer 0-9), phần dùng được cho task khác. |
| **Head** | Phần ra detection (layer 10+), chuyên cho box. |

---

> **Bình tĩnh — bạn đã nắm câu chuyện rồi 💪**
> Hỏi gì cũng quay về CÂU CỨU MẠNG ở đầu trang.
