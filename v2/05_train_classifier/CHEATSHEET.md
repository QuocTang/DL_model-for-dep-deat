# 📋 CHEAT SHEET — Bước 5 (in 2-3 trang mang theo)

## 🆘 CÂU CỨU MẠNG (học thuộc, bí thì đọc)

> **"Bước 5 dùng XGBoost — 450 cây quyết định, mỗi cây sửa lỗi cây trước — học vector 512 số từ Bước 4. Trên test set: accuracy 65%, macro F1 63%. Class hiếm `la-khoe` được giải cứu, đạt 80% recall (so với 0% của YOLO)."**

---

## 📖 Câu chuyện 1 phút (analogy 450 bác sĩ)

- **Bước 4**: mỗi ảnh → 512 con số.
- **Bước 5**: 450 "bác sĩ" cùng học để chẩn đoán bệnh từ 512 số đó.

```
Bác sĩ 1   chẩn đoán → SAI 30%
Bác sĩ 2   chỉ tập trung sửa 30% bác sĩ 1 sai → SAI 15%
Bác sĩ 3   sửa cái bác sĩ 2 còn sai → SAI 8%
...
Bác sĩ 450 hầu như không sai
```

→ **1 bác sĩ = 1 cây quyết định**. **"Boost" = bù lỗi, sửa dần**.

---

## 🔧 File `train_ml_features.py` làm 5 việc

1. Đọc 3 file CSV (`train/valid/test_features.csv`) từ Bước 4.
2. Chuẩn bị label (đảm bảo class id liên tục `0..5`).
3. Tạo XGBoost 450 cây + trọng số class hiếm (`sample_weight`).
4. Train: 450 cây học từ data, mỗi cây sửa lỗi cây trước.
5. Predict trên test → tính Accuracy/F1 → vẽ confusion matrix → lưu JSON.

→ **Output**: `xgb_model.joblib` + `confusion_matrix_test.png` + `results.json`.

---

## 🔢 3 số PHẢI nhớ thuộc

| Số | Câu nói |
|---|---|
| **450 cây** | "Em dùng 450 cây quyết định trong XGBoost." |
| **Test Acc 65%, Macro F1 63%** | "Trên test set, accuracy 65%, macro F1 63%." |
| **80% recall la-khoe** | "Class hiếm `la-khoe` đạt 80% recall — cứu class mà YOLO bỏ sót (0%)." |

---

## ❓ 8 câu Q&A có sẵn

**"XGBoost là gì?"**
> Thuật toán ML gồm 450 cây quyết định tuần tự, cây sau sửa lỗi cây trước. Phổ biến cho data dạng bảng.

**"Tại sao chọn XGBoost mà không phải CNN?"**
> CNN cần GPU + nhiều data. Em ~2700 ảnh + chạy CPU → XGBoost trên feature deep ổn hơn, nhanh hơn.

**"Tại sao 450 cây?"**
> Setting kinh nghiệm cho data ~2000 sample. Nhiều hơn dễ overfit + chậm. Ít hơn underfit.

**"`sample_weight` là gì?"**
> Trọng số mỗi sample. Class hiếm `la-khoe` (201 ảnh) có weight cao gấp ~3.8× class lớn `chay-la` (778 ảnh) để model chú ý.

**"Confusion matrix đọc sao?"**
> Bảng N×N: hàng = class **THẬT**, cột = class **ĐOÁN**. Đường chéo = đoán đúng. Off-diagonal đậm = nhầm nhiều.

**"Macro F1 khác Weighted F1?"**
> Macro = trung bình F1 của 6 class, **không quan tâm class lớn nhỏ**. Weighted = có trọng số theo số sample. Khi imbalance, **dùng Macro**.

**"Tại sao YOLO không classify luôn?"**
> YOLO mạnh ở **vị trí**, yếu ở classify cả ảnh. Combo YOLO (mắt) + XGBoost (não) tốt hơn dùng riêng.

**"Decision tree hoạt động sao?"**
> Chuỗi câu hỏi đi xuống lá: "Feature_42 > 0.3?" → có/không → câu tiếp → ra class.

---

## 🎛 5 HYPERPARAMETER chính (analogy "450 bác sĩ")

| # | Hyperparam | Code | Analogy | ↑ Tăng | ↓ Giảm |
|---|---|---|---|---|---|
| 1 | `n_estimators` | **450** | Số bác sĩ trong nhóm | Mạnh, dễ overfit, chậm | Yếu, underfit |
| 2 | `max_depth` | **8** | 1 bác sĩ hỏi tối đa 8 câu | Thông minh, dễ overfit | Đơn giản, underfit |
| 3 | `learning_rate` | **0.06** | Bác sĩ sau "tin" bác sĩ trước 6% | Học nhanh, dễ vọt | Chậm, ổn định |
| 4 | `subsample` | **0.85** | Mỗi bác sĩ xem 85% **bệnh án** | Trùng lặp, mất đa dạng | Quá ngẫu nhiên |
| 5 | `colsample_bytree` | **0.85** | Mỗi bác sĩ xem 85% **triệu chứng** | Trùng lặp | Quá ngẫu nhiên |

### Câu giải thích cho từng cái (nếu thầy hỏi sâu)

**`n_estimators=450`**
> 450 là sweet spot cho data ~2000 sample. Chọn theo kinh nghiệm.

**`max_depth=8`**
> Mỗi cây hỏi tối đa 8 câu trước khi quyết định. Sâu hơn dễ học thuộc lòng (overfit).

**`learning_rate=0.06`**
> Mỗi cây bù **6%** lỗi cây trước → học từ từ, không vọt qua optimum. Như đi cầu thang bước nhỏ → chắc.

**`subsample=0.85`**
> Mỗi cây xem 85% **dòng** data (15% giấu ngẫu nhiên) → 450 cây đa dạng góc nhìn.

**`colsample_bytree=0.85`**
> Mỗi cây xem 85% **cột** feature → tránh cả 450 cây tin vào cùng vài feature "vàng".

### 4 hyperparam phụ (để mặc định)

| Hyperparam | Code | Ý nghĩa |
|---|---|---|
| `objective` | `"multi:softprob"` | Multi-class, ra xác suất |
| `eval_metric` | `"mlogloss"` | Hàm sai số multi-class |
| `random_state` | `42` | Hạt giống, chạy lại ra y hệt |
| `n_jobs` | `-1` | Dùng tất cả CPU core |

---

## ⚠️ ĐỪNG NÓI NHẦM

- ❌ "XGBoost là deep learning" → ✅ XGBoost là **gradient boosting**, KHÔNG có neural network.
- ❌ "`sample_weight` làm cho data balanced" → ✅ Data **vẫn imbalance**, chỉ có **loss được nhân khác** để model chú ý class hiếm.
- ❌ "Accuracy 65% là kém" → ✅ Random 6 class = 16.7% → mình **gấp 4× random**, OK cho smoke-test.
- ❌ "`others` là noise" → ✅ `others` là class chính thức (gộp 10 class hiếm), F1 thấp vì đa dạng.
- ❌ "max_depth là số cây" → ✅ `max_depth` là độ sâu **1 cây**, `n_estimators` mới là số cây.

---

## 🎯 3 LOẠI MODEL (XGBoost vs RF vs LightGBM)

| | XGBoost ⭐ | RandomForest | LightGBM |
|---|---|---|---|
| Loại ensemble | **Boosting** (cây tuần tự, sửa lỗi nhau) | **Bagging** (cây độc lập, vote) | **Boosting** (giống XGB) |
| Train song song? | ⚠️ Một phần | ✅ Hoàn toàn | ⚠️ Một phần |
| Mạnh hơn | ✅ Pattern phức tạp | OK baseline | ✅ Data lớn (>10k) |
| Code mặc định | **xgb** | rf | lgbm |

→ Project mình dùng **XGBoost** (default). Có thể đổi qua `--model rf` hoặc `--model lgbm`.

---

## 📈 Confusion Matrix — đọc nhanh

```
                  pred→  sau-an  chay-la  than-thu  ...
true ↓
  sau-an              21       7        3       ...    ← hàng = class THẬT
  benh-chay-la         4      22        6       ...    ← cột = class ĐOÁN
  ...
```

**3 quy tắc**:
1. **Đường chéo** (true == pred) → đoán ĐÚNG. Muốn xanh đậm.
2. **Off-diagonal** đậm → nhầm nhiều. Vd `sau-an ↔ chay-la` nhầm 11 lần (cao nhất) → 2 bệnh có texture giống nhau.
3. **Hàng/cột nhạt** → class ít mẫu hoặc model bỏ qua.

**File**: `output/ml_models/confusion_matrix_test.png` (heatmap) + `.csv` (số chính xác).
**Mở PNG trên WSL**: `explorer.exe v2/output/ml_models/confusion_matrix_test.png`

---

## 📦 Pipeline tổng (slide cuối)

```
Bước 1 (refactor 15→6 class)
    ↓
Bước 2 (EDA: imbalance 18×)
    ↓
Bước 3 (Train YOLO: mAP50 = 0.295)
    ↓
Bước 4 (Bóp não YOLO → vector 512)
    ↓
Bước 5 (XGBoost: 65% Acc, 63% F1)   ← MÌNH ĐANG ĐÂY
    ↓                  ⭐ giải cứu la-khoe (0% → 80% recall)
Bước 6 (Inference end-to-end)
```

---

## 💡 Mini-định nghĩa Bước 5

| Từ | 1 câu |
|---|---|
| **Decision Tree** | Chuỗi câu hỏi đi xuống lá → ra class. |
| **XGBoost** | 450 cây tuần tự, mỗi cây sửa lỗi cây trước. |
| **Boosting vs Bagging** | Boosting (XGB) = cây học từ nhau. Bagging (RF) = cây độc lập, vote. |
| **`sample_weight` balanced** | Trọng số ngược tỉ lệ tần suất class — class hiếm weight cao. |
| **Overfit** | Học thuộc data train → test thấp. Tăng `gamma`, giảm `max_depth` để chống. |
| **Underfit** | Học chưa đủ. Tăng `n_estimators`, `max_depth`. |
| **Precision** | "Khi em nói sau-an, đúng %?" |
| **Recall** | "Trong sau-an thật, em bắt được %?" |
| **F1** | Trung bình hài hòa P và R. |
| **Macro F1** | Trung bình F1 của 6 class. **Quan trọng khi imbalance**. |
| **Accuracy** | Tổng đúng / tổng. |
| **Confusion Matrix** | Bảng N×N "ai nhầm với ai". |

---

## 🩺 Triệu chứng → Hành động (nếu thầy hỏi tune)

```
┌─ Train F1 cao, Test F1 thấp (OVERFIT) ──┬─ Hành động ────────┐
│                                         │ ↓ max_depth        │
│                                         │ ↑ gamma            │
│                                         │ ↓ learning_rate    │
├─────────────────────────────────────────┼────────────────────┤
│ Train F1 thấp, Test F1 thấp (UNDERFIT)  │ ↑ max_depth        │
│                                         │ ↑ n_estimators     │
├─────────────────────────────────────────┼────────────────────┤
│ Class hiếm yếu                          │ ↑ sample_weight    │
│                                         │ thử SMOTE          │
└─────────────────────────────────────────┴────────────────────┘
```

---

> **Bình tĩnh — bạn đã nắm câu chuyện rồi 💪**
> Hỏi gì cũng quay về CÂU CỨU MẠNG ở đầu trang.
