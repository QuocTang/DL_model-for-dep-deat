# 05 — Bản tóm tắt khi thuyết trình (~5 phút nói)

> File này chỉ chứa **ý chính để nói** + **số liệu cần nhớ** + **câu hỏi dự kiến**.
> Cần giải thích sâu? Đọc `NOTES.md` cùng thư mục.

---

## 🎯 Elevator pitch (30 giây — học thuộc)

> "Bước 5 là **bộ não** ăn vector 512 số từ Bước 4 và quyết định ảnh thuộc class nào. Mình dùng **XGBoost** — 450 cây quyết định học từ lỗi của nhau, kèm `sample_weight` để công bằng với class hiếm. Kết quả: **Test Accuracy 65%, Macro F1 63%**, và **giải cứu được class `la-khoe` mà YOLO bỏ sót hoàn toàn**."

---

## 🗣 5 ý chính khi trình bày

### 1. Tại sao chọn XGBoost? (45s)
- Vector 512 từ Bước 4 là **dữ liệu dạng bảng (tabular)** — XGBoost là vô địch trong mảng này.
- Đối thủ:
  - CNN classification head → nặng, cần GPU, dễ overfit với ~2000 sample.
  - Random Forest → đơn giản nhưng không "học từ lỗi".
- → XGBoost: nhẹ, chạy CPU 3 phút xong, kết quả thường top trong các cuộc thi Kaggle.

### 2. XGBoost hoạt động sao? (1 phút)
- **Cây 1** dự đoán → sai 30%.
- **Cây 2** chỉ tập trung sửa 30% sai đó → còn 15%.
- **Cây 3** sửa 15% còn lại → còn 8%.
- ... tiếp 450 cây, mỗi cây bù lỗi cây trước.
- "Boost" = **tăng cường, sửa dần**. Khác Random Forest: 500 cây độc lập, không học từ nhau.

### 3. Trick chống imbalance: `sample_weight` (45s)
- Class lớn (chay-la 778) >> class nhỏ (la-khoe 201) → model dễ "lười" đoán class lớn.
- Sklearn có `compute_sample_weight("balanced")` — class hiếm có weight CAO, class phổ biến weight THẤP.
- Pass vào `model.fit(X, y, sample_weight=...)` → loss của sample hiếm được nhân lớn → model phải chú ý.
- **Tác dụng**: la-khoe Recall = 80% dù chỉ có 10 ảnh test.

### 4. Đọc Confusion Matrix (1 phút)
```
                pred→  sau-an  chay-la  than-thu  ...
true ↓
  sau-an          21      7      3       ...   ← hàng = class THẬT
  chay-la          4     22      6       ...
  ...
```
- **Đường chéo** = đoán ĐÚNG → muốn xanh đậm.
- **Off-diagonal đậm** = nhầm nghiêm trọng. Vd `sau-an ↔ chay-la` nhầm 11 lần (cao nhất) → 2 bệnh có texture giống nhau.
- **Hàng nhạt** = class ít mẫu / model bỏ qua.

### 5. Kết quả + so sánh YOLO (45s)
- **Test Accuracy 65.2%, Macro F1 63.4%**.
- Class mạnh: la-khoe 73% F1, sau-an 71% F1.
- Class yếu: `others` 40% F1 (gộp 10 class linh tinh, không có pattern).
- **So với YOLO (Bước 3)**:
  - YOLO: la-khoe = 0% recall, dom-mat = 3% recall (bỏ sót class hiếm).
  - XGBoost: la-khoe = 80%, dom-mat = 70%.
  - → Combo YOLO + XGBoost > dùng riêng từng cái.

---

## 🔢 Số liệu cần nhớ

| Số | Ý nghĩa |
|---|---|
| **512** | chiều input (từ Bước 4) |
| **450** | số cây trong XGBoost |
| **8** | max_depth của 1 cây |
| **0.06** | learning_rate (bước học mỗi cây) |
| **0.85** | subsample + colsample_bytree (chống overfit) |
| **6** | số class output |
| **2359 / 227 / 112** | số ảnh train / valid / test |
| **65.2% / 63.4%** | Test Accuracy / Macro F1 |
| **80%** | la-khoe Recall — class hiếm được giải cứu |
| **18× → 3.9×** | imbalance giảm (từ Bước 4 sang Bước 5) |

---

## 🧠 4 thuật ngữ cốt lõi (giải thích bằng 1 câu)

| Thuật ngữ | Câu giải thích |
|---|---|
| **Gradient Boosting** | "Mỗi cây sau bù lỗi tổng các cây trước, sửa dần như học sinh sửa bài kiểm tra." |
| **sample_weight balanced** | "Đánh trọng số ngược tỉ lệ tần suất class — class hiếm có weight cao để model chú ý." |
| **Macro F1** | "Trung bình F1 của mọi class, không quan tâm class lớn nhỏ → quan trọng khi imbalance." |
| **Confusion Matrix** | "Bảng N×N: hàng = class thật, cột = class đoán; đường chéo = đúng, off-diagonal = sai." |

---

## ❓ Q&A dự kiến (đọc trước, đừng hoang mang)

**Q: Tại sao 450 cây mà không phải 100 hay 1000?**
> Setting kinh nghiệm cho data ~2000 sample: nhiều hơn → overfit + chậm, ít hơn → underfit. 450 + `learning_rate=0.06` là combo "chậm mà ổn". Nếu data lớn hơn (10k+ sample) thì có thể tăng lên 1000-2000 cây.

**Q: Sao không dùng deep CNN classifier cho mạnh?**
> Data mình có ~2000 sample → CNN dễ overfit, cần GPU + nhiều giờ train. XGBoost trên feature deep đã có sẵn (Bước 4) đạt kết quả tương đương, nhanh và chạy CPU. Đây là pattern phổ biến: "deep feature + classical classifier".

**Q: `sample_weight` khác `class_weight` thế nào?**
> Cùng mục đích chống imbalance. `class_weight` là setup ở model (vd RandomForest có sẵn), `sample_weight` là vector tính sẵn pass vào `fit()`. XGBoost không có `class_weight` → phải dùng `sample_weight`. Cả 2 cuối cùng đều nhân vào loss.

**Q: Tại sao Macro F1 thấp hơn Accuracy (63% vs 65%)?**
> Vì có class yếu (`others` F1 = 40%) kéo trung bình macro xuống. Accuracy chỉ tính tổng đúng/tổng → bị bias bởi class lớn. Khi imbalance, **Macro F1 là số đáng tin hơn**.

**Q: Tại sao map label phức tạp vậy (raw_to_model, model_to_raw)?**
> XGBoost yêu cầu label liên tục [0..n-1]. Bình thường data Bước 4 đã ổn. **Nhưng** nếu chạy `filter_top_classes.py` để chọn top-K, label cũ có thể `[0,1,3,4]` → cần remap về `[0,1,2,3]`. Code map 2 chiều để cuối cùng convert về raw label khi xuất predictions, khớp với CSV gốc.

**Q: Class `others` yếu (40% F1), có nên bỏ?**
> Có thể, dùng `04_features/filter_top_classes.py --top-k 5` để giữ 5 class mạnh. Trade-off: giảm coverage (ảnh others sẽ bị đoán nhầm sang 5 class kia). Tùy use case — nếu downstream chỉ care về 5 bệnh thì bỏ; nếu cần "fallback bucket" thì giữ.

**Q: So sánh XGBoost vs RandomForest vs LightGBM thì chọn cái nào?**
> Smoke-test này chỉ chạy XGBoost. Có thể chạy cả 3 với `--model rf/lgbm` rồi dùng `compare_models.py` gom bảng. Thường: XGBoost ≈ LightGBM > RandomForest về độ chính xác. LightGBM nhanh hơn 2-3× trên data lớn.

**Q: Heatmap PNG không thấy số trong ô?**
> Code đặt `annot=False` ở `train_ml_features.py:130`. Muốn thấy số: đổi `annot=True, fmt="d"` rồi train lại. Hoặc xem `confusion_matrix_test.csv` (cùng số liệu, format text).

---

## ⚠️ Đừng nói nhầm

- KHÔNG nói "XGBoost là deep learning" → XGBoost là **gradient boosting** trên cây quyết định, KHÔNG có neural network.
- KHÔNG nói "sample_weight làm cho data balance" → data vẫn imbalance, chỉ là **loss bị nhân khác** để model chú ý class hiếm.
- KHÔNG nói "Accuracy 65% là tốt/xấu" → cần ngữ cảnh: 6 class random = 16.7%, mình 65% = **gấp 4× random**, OK cho smoke-test.
- KHÔNG nói "class others là noise" → others là class chính thức (gộp 10 class hiếm ở Bước 1), chỉ là **F1 thấp vì đa dạng**.
- KHÔNG nói "test set bị leak" → train/valid/test split từ Bước 4 đã independent, model **chưa thấy** test khi train.

---

## 🎬 Kịch bản demo (nếu có máy)

```bash
cd v2
uv run 05_train_classifier/train_ml_features.py
```
→ Chạy ~3 phút trên CPU. Output cho khán giả thấy:
- `xgb_model.joblib` (~5 MB)
- `confusion_matrix_test.png` (mở `explorer.exe ...` trên WSL)
- `results.json` (highlight 3 dòng: accuracy, macro_f1, weighted_f1)

Hoặc demo nhanh `test_predictions.csv`:
```bash
head -5 v2/output/ml_models/test_predictions.csv
```
→ Cho khán giả thấy ảnh nào đúng/sai cụ thể.

---

## 📌 Kết nối với toàn pipeline (slide cuối)

```
Bước 1 (Refactor 15→6 class)
    ↓
Bước 2 (EDA: imbalance 18×)
    ↓
Bước 3 (Train YOLO detector — mAP50 = 0.295, miss class hiếm)
    ↓
Bước 4 (Bóp não YOLO → 512-dim vector cho mỗi ảnh)
    ↓
Bước 5 (Train XGBoost — Test Acc 65%, Macro F1 63%)   ← MÌNH ĐANG ĐỨNG ĐÂY
    ↓                  ⭐ giải cứu class hiếm la-khoe (0% → 80% recall)
Bước 6 (Inference end-to-end: ảnh → YOLO feature → XGBoost predict)
```

> Câu chốt: **"Bước 5 là khoảnh khắc các con số 512 chiều biến thành quyết định cuối cùng — và là lý do hệ thống không bỏ sót lá khỏe."**
