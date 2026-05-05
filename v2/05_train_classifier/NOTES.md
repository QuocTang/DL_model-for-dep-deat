# 05 — Train ML Classifier (XGBoost) — cho người mới

> Mục tiêu: Đọc xong file này, **không cần search ở đâu khác** vẫn hiểu Bước 5 làm gì, XGBoost hoạt động sao, các metric đọc thế nào.

---

## 1. Tại sao cần Bước 5?

### Câu chuyện
- **Bước 4** đã biến mỗi ảnh → **vector 512 số**.
- Giờ cần "ai đó" học: vector này → class nào (sâu ăn / cháy lá / thán thư / …).
- Lựa chọn:
  - Dùng **deep learning** (CNN classification head) → nặng, chậm, cần GPU.
  - Dùng **ML cổ điển** (XGBoost / RandomForest) → nhẹ, nhanh, chạy CPU vẫn ngon, kết quả thường rất tốt khi đã có feature deep.

→ Chọn **XGBoost**.

### Analogy
- YOLO (Bước 3) như **mắt** — nhìn ảnh thành số.
- XGBoost (Bước 5) như **não** — nhận số rồi quyết định "ảnh này thuộc class nào".

---

## 2. XGBoost là gì? (Phần lý thuyết, đọc 1 lần là đủ)

### 2.1. Decision Tree (cây quyết định)
Hình dung 1 cây hỏi-đáp:
```
Feature_42 > 0.3 ?
├── Có → Feature_117 > -0.5 ?
│        ├── Có → "sau-an"
│        └── Không → "benh-chay-la"
└── Không → Feature_8 > 1.2 ?
            ├── Có → "benh-than-thu"
            └── Không → "others"
```
Mỗi node hỏi 1 câu về 1 feature. Đi xuống lá → ra class.

**Ưu**: dễ hiểu. **Nhược**: 1 cây dễ sai (overfit hoặc quá đơn giản).

### 2.2. Random Forest = nhiều cây
Train **500 cây độc lập**, mỗi cây nhìn 1 phần dữ liệu khác nhau. Khi predict: 500 cây vote → lấy số đông.

→ Mạnh hơn, ít overfit. Nhưng các cây không "học từ lỗi nhau".

### 2.3. Gradient Boosting = cây học từ lỗi cây trước
Đây là tinh thần XGBoost:

```
Cây 1: dự đoán → sai 30%
Cây 2: chỉ tập trung sửa 30% sai đó → còn sai 15%
Cây 3: tập trung sửa 15% sai → còn sai 8%
...
Cây 450: sai rất ít
```

Mỗi cây sau **bù lỗi** của tổng các cây trước. **"Boost" = tăng cường, sửa dần**.

XGBoost (Extreme Gradient Boosting) là implementation **siêu nhanh, siêu mạnh** của ý tưởng này — gần như mặc định cho data dạng bảng (tabular) trong các cuộc thi Kaggle.

### 2.4. Pipeline mental model
```
Input: vector 512 số (feature_0, feature_1, ..., feature_511)
   ↓
[Cây 1] [Cây 2] [Cây 3] ... [Cây 450]   ← 450 cây
   ↓        ↓        ↓             ↓
Mỗi cây cho ra "điểm" cho từng class
   ↓
Tổng hợp 450 cây → xác suất cho 6 class
   ↓
Class nào xác suất cao nhất → predict
```

---

## 3. Các tham số quan trọng trong code

```python
XGBClassifier(
    n_estimators=450,      # số cây
    max_depth=8,           # độ sâu mỗi cây
    learning_rate=0.06,    # tốc độ học
    subsample=0.85,        # mỗi cây xem 85% sample
    colsample_bytree=0.85, # mỗi cây xem 85% feature
    objective="multi:softprob",
    eval_metric="mlogloss",
)
```

| Tham số | Ý nghĩa | Tăng → | Giảm → |
|---|---|---|---|
| `n_estimators=450` | Số cây | Học kỹ hơn, dễ overfit, chậm | Underfit |
| `max_depth=8` | Sâu max của cây | Phức tạp hơn, dễ overfit | Đơn giản hơn |
| `learning_rate=0.06` | Bước học mỗi cây | Học nhanh, dễ vọt qua optimum | Học chậm, cần nhiều cây |
| `subsample=0.85` | % sample cây thấy | (giữ 0.8-0.9) chống overfit | |
| `colsample_bytree=0.85` | % feature cây thấy | (giữ 0.8-0.9) chống overfit | |
| `objective=multi:softprob` | Bài toán nhiều class, ra xác suất | (cố định) | |
| `eval_metric=mlogloss` | Hàm đo lỗi multi-class | (cố định) | |

> Quy tắc vàng: `learning_rate` thấp + `n_estimators` cao = chậm nhưng ổn định. Đây là setting ổn cho data ~2000 sample.

---

## 4. Xử lý imbalance: `sample_weight`

Class lớn (chay-la 778 ảnh) >> class nhỏ (la-khoe 201 ảnh). Nếu không xử, model có xu hướng predict class lớn để được điểm cao.

**Trick**: Đánh **trọng số** cho mỗi sample:
- Sample của class hiếm → trọng số CAO (model chú ý nhiều hơn).
- Sample của class phổ biến → trọng số THẤP.

```python
sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
model.fit(X_train, y_train, sample_weight=sample_weight)
```

`balanced` công thức: `weight_i = n_total / (n_classes × n_samples_in_class_i)`. Class 100 sample sẽ có weight gấp 8× class 800 sample.

→ Model "công bằng" hơn với mọi class.

---

## 5. Pipeline 1 sample đi qua

```
Hàng 1 trong train_features.csv:
  feature_0 = 0.12
  feature_1 = -0.04
  ...
  feature_511 = -0.5
  label = 0 (sau-an)
                    ↓ map 0→0 (label đã liên tục)
                    ↓
  ┌───────────────────────────────────────┐
  │  XGBClassifier.fit(X_train, y_train)  │
  │   - 450 cây train tuần tự             │
  │   - Cây sau sửa lỗi cây trước         │
  │   - Có sample_weight chống imbalance  │
  └───────────────────────────────────────┘
                    ↓
  ┌───────────────────────────────────────┐
  │  Save: xgb_model.joblib + .json       │
  └───────────────────────────────────────┘
                    ↓
  Predict trên valid + test
                    ↓
  Tính metrics + confusion matrix
                    ↓
  Save report CSV + confusion PNG + JSON tóm tắt
```

---

## 6. Đọc Metrics: precision, recall, F1

### 6.1. Định nghĩa cơ bản (cho 1 class, ví dụ "sau-an")

Khi model dự đoán xong:
- **TP** (True Positive): predict "sau-an", thực tế đúng "sau-an"
- **FP** (False Positive): predict "sau-an", thực tế là class khác → **bắt nhầm**
- **FN** (False Negative): predict class khác, thực tế là "sau-an" → **bỏ sót**

| Metric | Công thức | Hỏi gì? |
|---|---|---|
| **Precision** | TP / (TP+FP) | Khi tôi nói "sau-an", tôi đúng bao nhiêu %? |
| **Recall** | TP / (TP+FN) | Trong các "sau-an" thật, tôi bắt được bao nhiêu %? |
| **F1** | 2·P·R/(P+R) | Trung bình hài hòa P và R |

### 6.2. Trade-off (như YOLO)
```
Precision cao, Recall thấp:  Đoán ít nhưng chắc      → bỏ sót
Precision thấp, Recall cao:  Đoán bừa, bắt được nhiều → bắt nhầm
F1 cao:                      Cân bằng                → ✅
```

### 6.3. Macro F1 vs Weighted F1
- **Macro F1** = trung bình F1 của 6 class, **không quan tâm class lớn nhỏ**.
  → Class hiếm có tiếng nói ngang class phổ biến. **Quan trọng khi imbalance**.
- **Weighted F1** = trung bình F1 có trọng số theo số sample.
  → Class lớn có tiếng nói nhiều hơn.

**Trong báo cáo, dùng Macro F1 vì class imbalance.**

---

## 7. Confusion Matrix — hướng dẫn đọc đầy đủ

### 7.1. Confusion matrix là gì?
Bảng vuông `N × N` (N = số class). Mỗi ô trả lời câu hỏi:
> "Trong số các ảnh thật sự là class A, model dự đoán bao nhiêu là class B?"

- **Hàng** (row) = **class THẬT** (sự thật)
- **Cột** (col) = **class DỰ ĐOÁN** (model nói gì)
- **Giá trị ô** = số mẫu

### 7.2. Cách mở file PNG (WSL → Windows)
File ở: `v2/output/ml_models/confusion_matrix_test.png`

```bash
# Trong WSL terminal
explorer.exe v2/output/ml_models/confusion_matrix_test.png
```
→ tự mở bằng app xem ảnh mặc định của Windows.

Hoặc xem CSV đi kèm (có số chính xác):
```bash
cat v2/output/ml_models/confusion_matrix_test.csv
```

### 7.3. Cách đọc HEATMAP (file PNG)

**Quy tắc nhìn màu**:
- 🟦 **Xanh đậm** = số lớn (ô có nhiều mẫu)
- ⬜ **Trắng/nhạt** = số nhỏ hoặc 0
- Bên phải có **thanh thước màu** (colorbar) — đối chiếu màu → ước lượng số

**Quy tắc đọc bố cục**:
```
                              ← Predicted →
                  sau-an  chay  than  dom   la-khoe  others
            ┌────────────────────────────────────────────┐
True    sau-an │ ⬛                                      │   ← hàng này nói: "31 ảnh sau-an thật, model đoán...?"
        chay   │     ⬛                                  │
        than   │          🟦                             │
        dom    │               🟦                        │   ← đường chéo (◢)
        la-khoe│                    🟦                   │
        others │                          🟦             │
            └────────────────────────────────────────────┘
                          ▲
                          │
                Đường chéo từ trên-trái xuống dưới-phải
                = ô "đoán ĐÚNG" (true == pred)
```

**3 vùng cần check**:
1. **Đường chéo càng đậm → model càng đúng**. Mục tiêu: cả 6 ô chéo đều xanh đậm.
2. **Off-diagonal đậm = nhầm lẫn nghiêm trọng**. Cần xem cụ thể class nào ↔ class nào nhầm.
3. **Hàng/cột nào toàn nhạt** → class đó ít mẫu hoặc model bỏ qua.

### 7.4. Đọc test set của mình (kèm số từ CSV):

```
                  sau-an  chay-la  than-thu  dom-mat  la-khoe  others   ┃ Tổng (true)
sau-an              21       7        3        0        0        0    ┃   31
benh-chay-la         4      22        6        0        0        2    ┃   34
benh-than-thu        2       2       11        1        0        0    ┃   16
benh-dom-mat-cua     0       1        0        7        0        2    ┃   10
la-khoe-binh-thg     0       0        0        1        8        1    ┃   10
others               1       1        0        1        4        4    ┃   11
─────────────────────────────────────────────────────────────────────
Tổng (pred)         28      33       20       10       12        9    ┃  112
```

### 7.5. 3 cách đọc thông tin từ matrix:

**Cách 1 — Đọc theo HÀNG → tính RECALL** (bắt được bao nhiêu của class này?)
```
Hàng "sau-an":  21 đúng / 31 thật = 67.7% Recall
                7 nhầm thành "chay-la"
                3 nhầm thành "than-thu"
→ Model BỎ SÓT 32% sau-an (đoán nhầm sang khác)
```

**Cách 2 — Đọc theo CỘT → tính PRECISION** (khi nói class này, đúng %?)
```
Cột "others":   model đoán "others" tổng 9 lần
                trong đó đúng 4
                → Precision = 4/9 = 44%
→ Khi model NÓI "others", chỉ đúng 44%
```

**Cách 3 — Tìm CẶP nhầm lẫn cao nhất** (off-diagonal đậm)
```
sau-an → chay-la: 7 lần   ┓
chay-la → sau-an: 4 lần   ┃ → cặp này lẫn 11 lần (cao nhất)
                          ┛   → 2 bệnh này LOOK ALIKE
chay-la → than-thu: 6 lần → bệnh nào texture giống bệnh nào
others → la-khoe: 4 lần   → "others" hay bị đoán nhầm thành la-khoe
```

### 7.6. Phát hiện từ test matrix:
- **`others` yếu nhất** (P=44%, R=36%) — vì class này gộp 10 class linh tinh, không có pattern thống nhất.
- **`la-khoe-binh-thuong` rất tốt** (R=80%) — dù chỉ có 10 sample test! → `sample_weight` đã giải cứu class hiếm.
- **`sau-an` ↔ `benh-chay-la`** dễ nhầm nhất (7+4=11 lần) → 2 bệnh này có texture giống nhau, cần thêm data hoặc augmentation.
- **`benh-than-thu`** Recall cao (69%) nhưng Precision thấp (55%) → model "đoán bừa" sang than-thu nhiều.

### 7.7. Confusion matrix vs Classification report
| | Confusion Matrix | Classification Report |
|---|---|---|
| Cho gì? | **Số mẫu** từng cặp (true, pred) | **Chỉ số** P/R/F1 đã tính sẵn |
| Khi nào dùng? | Tìm cặp nhầm lẫn, debug | Tóm tắt nhanh hiệu năng |
| File | `confusion_matrix_test.{png,csv}` | `classification_report_test.csv` |

**Workflow chuẩn**:
1. Xem **Classification Report** → biết class nào yếu (F1 thấp).
2. Xem **Confusion Matrix** → biết class đó nhầm với class nào.
3. Mở **`test_predictions.csv`** → tìm ảnh sai cụ thể, xem tận mắt.

### 7.8. ⚠️ Lưu ý: heatmap mặc định không hiện số
Code hiện tại có `annot=False` → chỉ thấy màu, không thấy số.

Muốn thấy số trong từng ô của PNG, sửa `train_ml_features.py:130`:
```python
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ...)
```
- `annot=True`: hiện số
- `fmt="d"`: định dạng integer (không phải float)

(Hiện tại có thể đọc số từ CSV — đỡ phải sửa code nếu chỉ làm 1 lần.)

---

## 8. Kết quả thực tế

```
Test Accuracy : 65.2%   (số mẫu đúng / tổng mẫu)
Test Macro F1 : 63.4%   (chú ý class hiếm)
Test Weighted F1: 65.0% (chú ý class lớn)
```

### Theo class (test):
| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| sau-an | 75% | 68% | 71% | 31 |
| benh-chay-la | 67% | 65% | 66% | 34 |
| benh-than-thu | 55% | 69% | 61% | 16 |
| benh-dom-mat-cua | 70% | 70% | 70% | 10 |
| **la-khoe-binh-thuong** | **67%** | **80%** | **73%** | 10 |
| others | 44% | 36% | 40% | 11 |

### So sánh YOLO (Bước 3) vs XGBoost (Bước 5)

> ⚠️ **Hai task khác nhau, không so trực tiếp được**:
> - YOLO: **detection** (vị trí + class) → metric mAP50
> - XGBoost: **classification cả ảnh** → metric Accuracy/F1

Nhưng có thể nhận xét chất lượng:
| | YOLO | XGBoost |
|---|---|---|
| Task | Detect (khó hơn) | Classify (dễ hơn) |
| `la-khoe` | 0% recall ❌ | 80% recall ✅ |
| `dom-mat-cua` | 3% recall ❌ | 70% recall ✅ |
| `sau-an` | 40% | 68% |

→ Bước 5 **giải cứu** class hiếm mà YOLO bỏ sót. Combo YOLO + XGBoost mạnh hơn dùng riêng.

---

## 9. Output (`v2/output/ml_models/`)

```
xgb_model.joblib              ⭐ model serialized (sklearn-compatible)
xgboost_model.json            ⭐ model native XGBoost format (cross-platform)
results.json                  ⭐ tóm tắt mọi metric + path

classification_report_valid.csv
classification_report_test.csv    ← bảng P/R/F1 từng class
confusion_matrix_valid.csv
confusion_matrix_test.csv
confusion_matrix_valid.png
confusion_matrix_test.png         ← heatmap đẹp để put vào báo cáo

test_predictions.csv          ← từng ảnh test: true vs pred (debug)
```

### `test_predictions.csv` để làm gì?
Mở ra xem ảnh nào model đoán sai → debug:
```csv
image_name, true_class, pred_class
IMG_001.jpg, sau-an, sau-an          ← đúng
IMG_023.jpg, benh-chay-la, sau-an    ← sai → mở ảnh xem tại sao
```

---

## 10. Lệnh chạy

### Mặc định (XGBoost)
```bash
cd v2
uv run 05_train_classifier/train_ml_features.py
```

### Thử model khác
```bash
uv run 05_train_classifier/train_ml_features.py --model rf      # Random Forest
uv run 05_train_classifier/train_ml_features.py --model lgbm    # LightGBM
```

### Tham số
- `--model {xgb,rf,lgbm}` — chọn thuật toán (default xgb)
- `--features-dir <path>` — đổi thư mục feature CSV
- `--output-dir <path>` — đổi nơi save model
- `--seed N` — random seed (default 42, để reproducible)

### Cài deps lần đầu
```bash
uv add scikit-learn xgboost seaborn joblib
```

---

## 11. Sau Bước 5 thì đi đâu?

→ **Bước 6**: Inference (suy luận) — cho 1 ảnh mới vào, dự đoán class.

Pipeline khi inference:
```
Ảnh mới → YOLO best.pt (Bước 3) → vector 512   (cùng cách Bước 4)
                                       ↓
                              XGBoost (Bước 5) → class predict
```

File: `v2/06_inference/cli.py`

---

## 12. Bonus: So sánh nhiều model (nếu muốn)

Có sẵn script `compare_models.py` chạy cả 3 (xgb / rf / lgbm) → bảng tổng hợp.

```bash
uv run 05_train_classifier/compare_models.py
```

(Chưa chạy ở smoke-test này, để sau khi muốn pick model tốt nhất cho production.)

---

## Tóm tắt 1 dòng

> Bước 5 = đưa **vector 512 số** từ Bước 4 cho **XGBoost** học → ra classifier với **Test Acc 65%, Macro F1 63%**. Class hiếm (la-khoe, dom-mat) được "giải cứu" so với YOLO.
