# 05 — Train ML Classifier (XGBoost) — cho người mới

> Mục tiêu: Đọc xong file này, **không cần search ở đâu khác** vẫn hiểu Bước 5 làm gì, **mỗi function trong `train_ml_features.py` viết để làm gì**, và đọc được Confusion Matrix lẫn các metric.

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

→ Tiếp theo: section 2 cho cái nhìn 30 giây về cấu trúc file, rồi section 3 đi từng đoạn code.

---

## 2. Bản đồ file: 1 cái nhìn tổng quan

Thư mục có 2 file Python:

```
05_train_classifier/
│
├── train_ml_features.py  (~282 dòng) ⭐ FILE CHÍNH
│   ├── 🛠 Setup (lines 1-32)
│   │   └── imports + parse_args
│   │
│   ├── 📂 Load + verify (lines 44-60)
│   │   ├── load_split()           ← đọc 1 CSV, validate columns
│   │   └── get_feature_columns()  ← lọc các cột feature_*
│   │
│   ├── 🧠 Build model (lines 63-101)
│   │   └── build_model()          ← chọn XGBoost / RandomForest / LightGBM
│   │
│   ├── 📊 Đánh giá (lines 104-147)
│   │   └── evaluate_split()       ← classification_report + confusion + heatmap
│   │
│   └── 🚀 Entrypoint (lines 150-281)
│       └── main()                 ← lắp ráp + map label + train + predict + save
│
└── compare_models.py     (~74 dòng)
    └── main()                     ← gom results.json của nhiều model → bảng so sánh
```

→ Hiểu **`main()` của train_ml_features.py** là hiểu 90% Bước 5. Section 3 đi qua đúng theo dòng chảy.

---

## 3. Đi qua code theo dòng chảy của 1 lần train

Phần này theo trình tự *thực tế khi script chạy*: load CSV → map label → build model → train với sample_weight → save → predict → evaluate → xuất file.

Mỗi mục có 3 phần cố định: **Concept** (tại sao), **Code** (snippet quan trọng), **Đọc code này thế nào** (giải thích).

---

### 3.1. Load 3 CSV + verify cùng số cột feature

**Concept**

Bước 4 đã tạo `train_features.csv` (2359 dòng), `valid_features.csv` (227 dòng), `test_features.csv` (112 dòng) trong `output/extracted_features/`. Mỗi file có **515 cột** = 512 feature + label + class_name + image_name.

Mình cần đảm bảo cả 3 split có **cùng tập cột feature** — nếu lệch thì model train trên chiều này không predict được trên chiều khác.

**Code** (`train_ml_features.py:44-60, 156-164`)
```python
def load_split(csv_path):
    df = pd.read_csv(csv_path)
    required = {"label", "class_name", "image_name"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns: {required}")
    return df

def get_feature_columns(df):
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    return feat_cols

# Trong main:
train_df = load_split(features_dir / "train_features.csv")
valid_df = load_split(features_dir / "valid_features.csv")
test_df  = load_split(features_dir / "test_features.csv")

feat_cols = get_feature_columns(train_df)
for name, split_df in [("valid", valid_df), ("test", test_df)]:
    missing = [c for c in feat_cols if c not in split_df.columns]
    if missing:
        raise ValueError(f"{name} split missing feature columns; count={len(missing)}")
```

**Đọc code này thế nào**
- `load_split` chỉ là `pd.read_csv` + check 3 cột bắt buộc (`label`, `class_name`, `image_name`). Đây là **fail-fast** — phát hiện CSV hỏng ngay lúc đầu thay vì chết giữa training.
- `get_feature_columns` lọc cột bắt đầu bằng `"feature_"` (vd `feature_0, feature_1, ..., feature_511`) — tách ra khỏi 3 cột metadata.
- Vòng for sau đó **so cột train với valid/test**: nếu thiếu cột nào (vd file CSV cũ tạo bằng layer khác) sẽ raise sớm.

---

### 3.2. Map label cũ → label liên tục [0..n-1]

**Concept**

XGBoost yêu cầu label phải **liên tục** từ 0 đến n_classes-1, không có gap. Vấn đề:
- Sau Bước 4 thường ổn (label = 0..5 cho 6 class).
- **Nhưng** nếu dùng `filter_top_classes.py` để chọn top-K, label cũ có thể là `[0, 1, 3, 4]` (thiếu 2, 5) → XGBoost lỗi.

→ Script tự xây map `raw_to_model` để chuyển label về liên tục. Đồng thời nhớ `model_to_raw` để khi xuất predictions còn convert ngược lại.

**Code** (`train_ml_features.py:166-184`)
```python
class_map = (                          # {0: "sau-an", 1: "benh-chay-la", ...}
    train_df[["label", "class_name"]]
    .drop_duplicates()
    .sort_values("label")
    .set_index("label")["class_name"]
    .to_dict()
)
raw_train_labels = sorted(class_map.keys())          # [0, 1, 3, 4] nếu có gap
class_names = [class_map[k] for k in raw_train_labels]

raw_to_model = {raw: idx for idx, raw in enumerate(raw_train_labels)}  # {0:0, 1:1, 3:2, 4:3}
model_to_raw = {idx: raw for raw, idx in raw_to_model.items()}

def map_labels(raw_labels):
    mapped = np.array([raw_to_model.get(int(v), -1) for v in raw_labels], dtype=np.int64)
    known_mask = mapped >= 0
    return mapped, known_mask
```

**Đọc code này thế nào**
- `class_map` dict: `{label_id: class_name}` — đọc từ training CSV (nguồn chân lý duy nhất). KHÔNG xây từ valid/test vì có thể thiếu class.
- `enumerate(raw_train_labels)` đảm bảo idx mới luôn liên tục `0, 1, 2, ...` bất kể raw có gap hay không.
- `map_labels` trả về thêm `known_mask` — đánh dấu sample nào có label nằm trong train. Test set có thể có class không thấy trong train → bị `-1` → mask out để không tính metric trên những sample này.
- Hàm `map_labels` được khai báo **trong** `main()` (closure) để truy cập `raw_to_model` mà không cần truyền tham số.

---

### 3.3. Chọn model: XGBoost / RandomForest / LightGBM

**Concept** — *3 thuật toán tabular phổ biến*

#### 3.3.1. Decision Tree (cây quyết định)
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
**Ưu**: dễ hiểu. **Nhược**: 1 cây dễ sai (overfit hoặc quá đơn giản).

#### 3.3.2. Random Forest = nhiều cây độc lập (Bagging)
Train **500 cây độc lập**, mỗi cây nhìn 1 phần dữ liệu khác nhau. Predict: vote.
→ Mạnh hơn, ít overfit. Nhưng các cây không "học từ lỗi nhau".

#### 3.3.3. Gradient Boosting = cây học từ lỗi cây trước (Boosting)
Tinh thần XGBoost / LightGBM:
```
Cây 1: dự đoán → sai 30%
Cây 2: tập trung sửa 30% sai đó → còn sai 15%
Cây 3: tập trung sửa 15% sai → còn sai 8%
...
Cây 450: sai rất ít
```
Mỗi cây sau **bù lỗi** tổng các cây trước. **"Boost" = tăng cường, sửa dần**.

→ XGBoost là implementation siêu nhanh, gần như mặc định cho data tabular ở Kaggle.

**Code** (`train_ml_features.py:63-101`)
```python
def build_model(model_name, seed):
    if model_name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=450, max_depth=8, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=seed, n_jobs=-1,
        )
    if model_name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=450, learning_rate=0.06, num_leaves=63,
            subsample=0.85, colsample_bytree=0.85,
            objective="multiclass", random_state=seed, n_jobs=-1, verbosity=-1,
        )
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=500, class_weight="balanced_subsample",
        random_state=seed, n_jobs=-1,
    )
```

**Đọc code này thế nào**
- **Lazy import** (`from xgboost import ...` trong if): chỉ load thư viện cần dùng → người chỉ chạy RF không cần cài XGBoost.
- `random_state=seed` cho **reproducibility**: chạy lại ra kết quả y hệt.
- `n_jobs=-1` dùng tất cả CPU core có sẵn.
- RandomForest dùng `class_weight="balanced_subsample"` (built-in) — không cần pass `sample_weight` thủ công như 2 model còn lại.

**Bảng tham số XGBoost — học thuộc**
| Tham số | Giá trị | Tăng → | Giảm → |
|---|---|---|---|
| `n_estimators=450` | Số cây | Học kỹ hơn, dễ overfit, chậm | Underfit |
| `max_depth=8` | Sâu max của cây | Phức tạp, dễ overfit | Đơn giản |
| `learning_rate=0.06` | Bước học mỗi cây | Học nhanh, dễ vọt qua optimum | Học chậm, cần nhiều cây |
| `subsample=0.85` | % sample mỗi cây thấy | (giữ 0.8-0.9) chống overfit | |
| `colsample_bytree=0.85` | % feature mỗi cây thấy | (giữ 0.8-0.9) chống overfit | |
| `objective=multi:softprob` | Multi-class, ra xác suất | (cố định) | |
| `eval_metric=mlogloss` | Hàm lỗi multi-class | (cố định) | |

> Quy tắc vàng: `learning_rate` thấp + `n_estimators` cao = chậm nhưng ổn định. Setting trên ổn cho data ~2000 sample.

---

### 3.4. Xử lý imbalance: `sample_weight`

**Concept**

Class lớn (chay-la 778 ảnh) >> class nhỏ (la-khoe 201 ảnh). Nếu không xử, model có xu hướng **predict class lớn để được điểm cao** (kiểu "cứ đoán A là 30% đúng").

**Trick**: Đánh trọng số mỗi sample:
- Sample của class hiếm → trọng số CAO (model chú ý nhiều hơn).
- Sample của class phổ biến → trọng số THẤP.

Công thức `balanced`: `weight_i = n_total / (n_classes × n_samples_in_class_i)`.
→ Class 100 sample sẽ có weight gấp ~8× class 800 sample.

**Code** (`train_ml_features.py:200-208`)
```python
model = build_model(args.model, args.seed)

sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
fit_kwargs = {"sample_weight": sample_weight}

if args.model == "xgb":
    fit_kwargs["verbose"] = False

model.fit(X_train, y_train, **fit_kwargs)
```

**Đọc code này thế nào**
- `compute_sample_weight` (sklearn) tự tính weight theo tần suất class trong `y_train` — không cần code thủ công.
- `fit_kwargs` đóng gói kwargs để dễ thêm tùy theo model. XGBoost có thêm `verbose=False` để bớt log spam khi train.
- `model.fit(X_train, y_train, sample_weight=...)` — sample_weight nhân vào loss của mỗi sample. Class hiếm sai → loss bị nhân lớn → model phải sửa.
- RandomForest bỏ qua `sample_weight` ở đây vì nó dùng `class_weight="balanced_subsample"` setup từ `build_model` rồi (cách khác đạt cùng mục đích).

---

### 3.5. Save model — 2 format

**Concept**

Save model cho phép load lại sau (Bước 6 inference) mà không cần train lại. 2 format:
- **`joblib`** (sklearn-compatible): pickle nhị phân — load nhanh trong Python, **gắn với version sklearn/XGBoost** nên có thể vỡ khi đổi máy.
- **`xgboost_model.json`** (XGBoost native): JSON cross-platform — đọc được bởi mọi binding XGBoost (Python/R/JVM/C++), bền version hơn.

→ Save **cả 2** cho XGBoost. RF và LGBM chỉ save joblib (chưa có nhu cầu cross-platform).

**Code** (`train_ml_features.py:210-216`)
```python
model_joblib_path = output_dir / f"{args.model}_model.joblib"
joblib.dump(model, model_joblib_path)

xgb_native_path = None
if args.model == "xgb":
    xgb_native_path = output_dir / "xgboost_model.json"
    model.save_model(str(xgb_native_path))
```

**Đọc code này thế nào**
- `joblib.dump` viết binary, file ~5-10 MB cho XGBoost 450 cây.
- `model.save_model(str(...))` là API của XGBoost (sklearn không có) → format JSON readable.
- `args.model` dùng làm prefix tên file (`xgb_model.joblib`, `rf_model.joblib`, `lgbm_model.joblib`) → cùng output dir vẫn không đè nhau khi chạy 3 model so sánh.

---

### 3.6. Predict + evaluate trên valid và test

**Concept**

Sau train, dùng `predict()` chạy model trên valid và test (model **chưa thấy** trong training).
- **Valid** = "kỳ thi giữa kỳ" — dùng nếu cần tinh chỉnh hyperparameter (ở smoke-test này không tune nên valid chỉ để báo cáo).
- **Test** = "kỳ thi cuối" — số chính thức báo cáo.

Với mỗi split, tính 3 thứ:
1. `classification_report`: bảng P/R/F1 từng class.
2. `confusion_matrix`: ma trận N×N "ai nhầm với ai".
3. Heatmap PNG để put vào báo cáo.

**Code** (`train_ml_features.py:104-147, 218-226`)
```python
def evaluate_split(split_name, y_true, y_pred, class_names, output_dir):
    labels = list(range(len(class_names)))
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    pd.DataFrame(report_dict).T.to_csv(output_dir / f"classification_report_{split_name}.csv")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        output_dir / f"confusion_matrix_{split_name}.csv"
    )

    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix ({split_name})")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{split_name}.png", dpi=160)
    plt.close()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        # ...
    }

# Trong main:
y_pred_valid = model.predict(X_valid)
y_pred_test  = model.predict(X_test)
metrics_valid = evaluate_split("valid", y_valid[valid_mask], y_pred_valid[valid_mask], ...)
metrics_test  = evaluate_split("test",  y_test[test_mask],  y_pred_test[test_mask],   ...)
```

**Đọc code này thế nào**
- `zero_division=0` quan trọng: nếu 1 class hoàn toàn không được predict, P sẽ là 0/0 → mặc định raise warning. `=0` cho ra giá trị 0 thay vì lỗi.
- `output_dict=True` → `classification_report` trả dict thay vì string → dễ chuyển sang DataFrame và save CSV.
- `[valid_mask]` / `[test_mask]` lọc bỏ sample có label "lạ" (raw không có trong train) — đã xử ở section 3.2.
- `annot=False` → heatmap **không hiện số trong ô** (chỉ hiện màu). Số xem ở CSV. Muốn thấy số trong PNG, đổi thành `annot=True, fmt="d"`.
- 2 metric chính: `accuracy` (đơn giản, dễ hiểu) và `macro_f1` (chú ý class hiếm — quan trọng khi imbalance).

---

### 3.7. Save predictions CSV + summary JSON

**Concept**

2 file cuối:
- **`test_predictions.csv`**: mỗi dòng 1 ảnh test với `true_class` và `pred_class` → mở ra debug ảnh nào sai để xem tận mắt.
- **`results.json`**: tóm tắt mọi thứ (model name, metrics, paths, class mapping…) → để `compare_models.py` đọc lại.

**Code** (`train_ml_features.py:231-264`)
```python
pred_df = pd.DataFrame({
    "image_name": test_df["image_name"],
    "true_label_raw": y_test_raw,
    "pred_label_raw": [model_to_raw[int(x)] for x in y_pred_test],
    "true_class": [class_map.get(int(x), str(x)) for x in y_test_raw],
    "pred_class": [class_map.get(model_to_raw[int(x)], str(int(x))) for x in y_pred_test],
    "is_seen_in_train": test_mask,
})
pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

summary = {
    "model": args.model, "seed": args.seed,
    "n_features": len(feat_cols),
    "class_mapping": class_map,
    "label_mapping_raw_to_model": raw_to_model,
    "valid": metrics_valid, "test": metrics_test,
    "test_predictions_csv": str(pred_csv),
    "model_file": str(model_joblib_path),
    # ...
}
with open(output_dir / "results.json", "w") as f:
    json.dump(summary, f, indent=2)
```

**Đọc code này thế nào**
- `model_to_raw[int(x)]` convert label model (liên tục 0..n-1) **ngược về** label gốc (có thể có gap) → predictions CSV dùng raw label cho khớp với CSV của Bước 4.
- `class_map.get(..., str(x))` fallback ra string nếu không tìm thấy — guard cho trường hợp test có class lạ.
- `is_seen_in_train` cờ True/False để filter khi audit predictions: sample False = label test không thấy trong train, không nên blame model.
- `results.json` là **single source of truth** để báo cáo và để `compare_models.py` đọc tổng hợp.

---

## 4. Đọc Metrics: precision, recall, F1

### 4.1. Định nghĩa cơ bản (cho 1 class, ví dụ "sau-an")

Khi model dự đoán xong:
- **TP** (True Positive): predict "sau-an", thực tế đúng "sau-an"
- **FP** (False Positive): predict "sau-an", thực tế là class khác → **bắt nhầm**
- **FN** (False Negative): predict class khác, thực tế là "sau-an" → **bỏ sót**

| Metric | Công thức | Hỏi gì? |
|---|---|---|
| **Precision** | TP / (TP+FP) | Khi tôi nói "sau-an", tôi đúng bao nhiêu %? |
| **Recall** | TP / (TP+FN) | Trong các "sau-an" thật, tôi bắt được bao nhiêu %? |
| **F1** | 2·P·R/(P+R) | Trung bình hài hòa P và R |

### 4.2. Trade-off (như YOLO)
```
Precision cao, Recall thấp:  Đoán ít nhưng chắc      → bỏ sót
Precision thấp, Recall cao:  Đoán bừa, bắt được nhiều → bắt nhầm
F1 cao:                      Cân bằng                → ✅
```

### 4.3. Macro F1 vs Weighted F1
- **Macro F1** = trung bình F1 của 6 class, **không quan tâm class lớn nhỏ**.
  → Class hiếm có tiếng nói ngang class phổ biến. **Quan trọng khi imbalance**.
- **Weighted F1** = trung bình F1 có trọng số theo số sample.
  → Class lớn có tiếng nói nhiều hơn.

**Trong báo cáo, dùng Macro F1 vì class imbalance.**

---

## 5. Confusion Matrix — hướng dẫn đọc đầy đủ

### 5.1. Confusion matrix là gì?
Bảng vuông `N × N` (N = số class). Mỗi ô trả lời câu hỏi:
> "Trong số các ảnh thật sự là class A, model dự đoán bao nhiêu là class B?"

- **Hàng** (row) = **class THẬT** (sự thật)
- **Cột** (col) = **class DỰ ĐOÁN** (model nói gì)
- **Giá trị ô** = số mẫu

### 5.2. Cách mở file PNG (WSL → Windows)
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

### 5.3. Cách đọc HEATMAP (file PNG)

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

### 5.4. Đọc test set của mình (kèm số từ CSV):

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

### 5.5. 3 cách đọc thông tin từ matrix

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

### 5.6. Phát hiện từ test matrix:
- **`others` yếu nhất** (P=44%, R=36%) — vì class này gộp 10 class linh tinh, không có pattern thống nhất.
- **`la-khoe-binh-thuong` rất tốt** (R=80%) — dù chỉ có 10 sample test! → `sample_weight` đã giải cứu class hiếm.
- **`sau-an` ↔ `benh-chay-la`** dễ nhầm nhất (7+4=11 lần) → 2 bệnh này có texture giống nhau, cần thêm data hoặc augmentation.
- **`benh-than-thu`** Recall cao (69%) nhưng Precision thấp (55%) → model "đoán bừa" sang than-thu nhiều.

### 5.7. Confusion matrix vs Classification report
| | Confusion Matrix | Classification Report |
|---|---|---|
| Cho gì? | **Số mẫu** từng cặp (true, pred) | **Chỉ số** P/R/F1 đã tính sẵn |
| Khi nào dùng? | Tìm cặp nhầm lẫn, debug | Tóm tắt nhanh hiệu năng |
| File | `confusion_matrix_test.{png,csv}` | `classification_report_test.csv` |

**Workflow chuẩn**:
1. Xem **Classification Report** → biết class nào yếu (F1 thấp).
2. Xem **Confusion Matrix** → biết class đó nhầm với class nào.
3. Mở **`test_predictions.csv`** → tìm ảnh sai cụ thể, xem tận mắt.

### 5.8. ⚠️ Lưu ý: heatmap mặc định không hiện số
Code hiện tại có `annot=False` (`train_ml_features.py:130`) → chỉ thấy màu, không thấy số.

Muốn thấy số trong từng ô của PNG, sửa:
```python
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ...)
```
- `annot=True`: hiện số
- `fmt="d"`: định dạng integer (không phải float)

(Hiện tại có thể đọc số từ CSV — đỡ phải sửa code nếu chỉ làm 1 lần.)

---

## 6. Kết quả thực tế

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

## 7. Output (`v2/output/ml_models/`)

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

## 8. Lệnh chạy

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
# (Optional) uv add lightgbm  — nếu muốn thử LGBM
```

---

## 9. Sau Bước 5 thì đi đâu?

→ **Bước 6**: Inference (suy luận) — cho 1 ảnh mới vào, dự đoán class.

Pipeline khi inference:
```
Ảnh mới → YOLO best.pt (Bước 3) → vector 512   (cùng cách Bước 4)
                                       ↓
                              XGBoost (Bước 5) → class predict
```

File: `v2/06_inference/cli.py`

---

## 10. Bonus: `compare_models.py` — so sánh nhiều model

### Mục đích
Sau khi chạy `train_ml_features.py` với nhiều `--model` khác nhau (xgb / rf / lgbm), mỗi run tạo `results.json` riêng. Script này **gom tất cả** thành 1 bảng so sánh.

### Code (`compare_models.py:24-43`)
```python
for result_path in sorted(args.comparison_dir.glob("*/results.json")):
    with open(result_path) as f:
        data = json.load(f)
    rows.append({
        "model": data["model"],
        "valid_accuracy": data["valid"]["accuracy"],
        "valid_macro_f1": data["valid"]["macro_f1"],
        "test_accuracy":  data["test"]["accuracy"],
        "test_macro_f1":  data["test"]["macro_f1"],
        "model_file":     data.get("model_file"),
        # ...
    })
df = pd.DataFrame(rows).sort_values(["test_macro_f1", "test_accuracy"], ascending=False)
```

### Lệnh chạy
```bash
# Trước tiên chạy 3 model vào 3 folder riêng
uv run 05_train_classifier/train_ml_features.py --model xgb  --output-dir output/model_comparison/xgb
uv run 05_train_classifier/train_ml_features.py --model rf   --output-dir output/model_comparison/rf
uv run 05_train_classifier/train_ml_features.py --model lgbm --output-dir output/model_comparison/lgbm

# Rồi gom lại
uv run 05_train_classifier/compare_models.py --comparison-dir output/model_comparison
```

→ Ra `model_comparison_summary.{csv,md}` xếp model theo `test_macro_f1` giảm dần.

(Chưa chạy ở smoke-test này, để sau khi muốn pick model tốt nhất cho production.)

---

## Tóm tắt 1 dòng

> Bước 5 = đưa **vector 512 số** từ Bước 4 cho **XGBoost** (450 cây boosting) học, có **sample_weight chống imbalance**, ra classifier với **Test Acc 65%, Macro F1 63%**. Trái tim file `train_ml_features.py` là `main()`: load CSV → map label liên tục → build_model → fit với sample_weight → predict → evaluate (P/R/F1 + confusion matrix) → save 2 format model + JSON tóm tắt. Class hiếm (la-khoe, dom-mat) được "giải cứu" so với YOLO.
