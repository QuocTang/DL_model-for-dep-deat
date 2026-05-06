# 04 — Deep Feature Extraction (cho người mới)

> Mục tiêu: Đọc xong file này, **không cần search ở đâu khác** vẫn hiểu Bước 4 làm gì, **mỗi function trong `extract_deep_features.py` viết để làm gì**, và tại sao lại viết theo cách đó.

---

## 1. Tại sao cần Bước 4?

### Câu chuyện
- **Bước 3** đã train xong YOLO → model biết "đây là lá bị sâu ăn", "đây là chấm mắt cua"…
- Nhưng YOLO làm **detection** (vẽ hộp + gán nhãn cho từng vật trong ảnh).
- Mình lại muốn **classification** (1 nhãn cho cả ảnh) bằng **XGBoost** (một thuật toán ML cổ điển — nhanh, dễ giải thích).
- Vấn đề: XGBoost không "nhìn" được ảnh thô. Nó chỉ ăn **vector số** (ví dụ `[0.12, -0.04, 1.3, ...]`).

→ **Bước 4 = cây cầu**: Biến mỗi ảnh thành 1 vector số (512 chiều) bằng cách "mượn não" của YOLO đã train.

### Analogy
Hãy tưởng tượng YOLO là một **nhà phê bình tranh** đã xem hàng ngàn bức tranh.
- Bước 3: dạy nó "đây là tranh Picasso, đây là Van Gogh".
- Bước 4: bảo nó "với mỗi bức tranh, hãy mô tả thành 512 con số đặc trưng nhất".
- Bước 5: đưa 512 số đó cho 1 học sinh khác (XGBoost) học "512 số này → tranh ai".

Tại sao không để YOLO làm hết? Vì:
- YOLO mạnh ở **vị trí object**, yếu ở **classify cả ảnh**.
- XGBoost rất mạnh ở classification trên vector. Combo = win.

→ Tiếp theo: section 2 cho cái nhìn 30 giây về cấu trúc file, rồi section 3 đi từng đoạn code.

---

## 2. Bản đồ file: 1 cái nhìn tổng quan

`extract_deep_features.py` (~445 dòng) chia làm 5 nhóm:

```
extract_deep_features.py
│
├── 🛠 Setup (lines 1-31)
│   ├── imports: cv2, numpy, torch, ultralytics.YOLO, ...
│   └── DEFAULT_*  ← lấy từ v2/config.py (model path, output dir, ...)
│
├── 🧠 YoloFeatureExtractor — class chính (lines 56-133)
│   ├── __init__()              ← load YOLO + cắm "camera" (hook) vào layer 4,6,9
│   ├── _register_hooks()       ← đặt camera tại layer được chỉ định
│   ├── _letterbox()            ← resize ảnh giữ tỉ lệ + pad xám
│   ├── extract_array()         ← forward 1 ảnh → vector 512
│   └── extract_one()           ← đọc file ảnh → gọi extract_array
│
├── 📂 Đọc nhãn / liệt kê ảnh (lines 136-199, 316-334)
│   ├── read_names()            ← class names từ data.yaml (YOLO)
│   ├── collect_image_paths()   ← liệt kê *.jpg/*.png trong split
│   ├── resolve_split_dir()     ← path trong data.yaml → path tuyệt đối
│   ├── find_split_dir()        ← tìm thư mục train/valid/test (TF format)
│   ├── load_tf_split_labels()  ← parse _annotations.csv
│   └── build_tf_class_mapping()← gom class từ tất cả split
│
├── 🔄 Loop trên 1 split (lines 202-313)
│   ├── extract_split()             ← format YOLO (.txt labels)
│   └── extract_split_tensorflow()  ← format TF (CSV labels)
│
└── 🚀 Entrypoint (lines 337-446)
    ├── parse_args()
    └── main()                  ← lắp ráp + auto-detect format + chạy 3 split
```

> Hiểu **5 method trong `YoloFeatureExtractor`** là hiểu 80% file này. Section 3 đi qua đúng theo dòng chảy của 1 ảnh.

---

## 3. Đi qua code theo dòng chảy của 1 ảnh

Phần này đi theo trình tự *thực tế khi script chạy*: nạp model → chuẩn bị 1 ảnh → forward → pool → đọc nhãn → loop nhiều ảnh → ghi CSV.

Mỗi mục có 3 phần cố định: **Concept** (tại sao), **Code** (snippet quan trọng), **Đọc code này thế nào** (giải thích).

---

### 3.1. Setup: load YOLO + cắm "camera" vào layer 4, 6, 9

**Concept**

Bình thường khi gọi `model(image)`, YOLO chạy hết từ đầu đến cuối, chỉ trả về **kết quả cuối** (detection — hộp + class). Mình muốn **chộp output ở giữa** (layer 4, 6, 9). Cách: gắn 1 **hook** — như đặt camera ở giữa pipeline để ghi lại output mà không phá code gốc của YOLO.

> Vì sao layer 4-9? CNN học từ thô đến mịn: layer 0-3 học cạnh/màu (quá thấp, ảnh nào cũng giống), layer 10+ chuyên cho detection (quá đặc thù). Layer 4-9 ở giữa = texture + hình nhỏ — vừa đủ phong phú để phân biệt bệnh, vừa đủ tổng quát để ML khác dùng được.

**Code** (`extract_deep_features.py:56-82`)
```python
class YoloFeatureExtractor:
    def __init__(self, model_path, layer_indices, input_size=640):
        self.layer_indices = layer_indices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(str(model_path))
        self.yolo.model.to(self.device)
        self.features: Dict[str, torch.Tensor] = {}    # buffer "chộp" output
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(name):
            def hook(module, inputs, output):
                self.features[name] = output.detach()  # chộp ở đây
            return hook
        layers = self.yolo.model.model
        for idx in self.layer_indices:
            layers[idx].register_forward_hook(get_hook(f"layer_{idx}"))
```

**Đọc code này thế nào**
- `self.features = {}` là buffer dạng dict, key là tên layer (`"layer_4"`, `"layer_6"`, `"layer_9"`), value là tensor output. Mỗi lần forward 1 ảnh, dict được điền 3 entry.
- `get_hook(name)` là **factory tạo hook** (closure). Hàm trong `register_forward_hook` chỉ nhận `(module, inputs, output)` — không nhận tên layer, nên cần closure để mỗi hook "nhớ" tên của mình. Đây là idiom Python phổ biến khi gắn callback có context.
- `output.detach()` cắt tensor khỏi computation graph (không tính gradient nữa) → tiết kiệm RAM. Mình chỉ inference, không train.
- Nếu `idx > max_idx` (layer ngoài model), code raise `ValueError` — guard để tránh silent bug.

---

### 3.2. Chuẩn bị 1 ảnh: letterbox về 320×320

**Concept**

Ảnh thực có nhiều tỉ lệ (1920×1080, 800×600…), nhưng model cần input vuông cố định. Có 2 cách:
- **Squish thuần** (`cv2.resize` → 320×320): nhanh nhưng ảnh bị méo (cái lá tròn → elip).
- **Letterbox** (resize giữ tỉ lệ + pad xám 114 các viền): không méo, hơi tốn pixel viền.

YOLO **được train với letterbox** → mình cũng phải letterbox lúc extract, không thì vector sẽ "lệch" so với phân bố mà model đã học.

**Code** (`extract_deep_features.py:84-100`)
```python
def _letterbox(self, image):
    h, w = image.shape[:2]
    ratio = self.input_size / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw, dh = self.input_size - new_w, self.input_size - new_h
    left, top = dw // 2, dh // 2
    return cv2.copyMakeBorder(
        resized, top, dh - top, left, dw - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
```

**Đọc code này thế nào**
- `ratio = input_size / max(h, w)` lấy cạnh dài làm chuẩn → ảnh 1920×1080 → ratio = 320/1920 ≈ 0.167 → resize thành 320×167 (giữ tỉ lệ).
- `dh, dw` là phần thiếu để đủ vuông 320. Pad **đều 2 bên** (`// 2`) để ảnh canh giữa.
- `value=(114, 114, 114)` là xám trung tính — convention của Ultralytics. **Không** đổi sang 0 (đen) hay 255 (trắng) vì model đã học pad đúng giá trị này.

---

### 3.3. Forward 1 ảnh + chộp 3 feature map

**Concept**

Output của 1 layer = một **stack các "ảnh xám" nhỏ** (gọi là feature map / activation). Ví dụ layer 4 cho ra `[64 × 40 × 40]`:
- 64 = số "kênh" (mỗi kênh phát hiện 1 loại pattern: cạnh dọc, vết tròn, đốm…).
- 40 × 40 = bản đồ vị trí (mỗi ô tương ứng 1 vùng 8×8 pixel của ảnh gốc).

Có thể xem như **64 tấm bản đồ nhiệt** đánh dấu "vùng nào của ảnh kích hoạt pattern này".

**Code** (`extract_deep_features.py:102-114`)
```python
def extract_array(self, image_rgb):
    image = self._letterbox(image_rgb)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # HWC→CHW, 0-1
    tensor = tensor.unsqueeze(0).to(self.device)                       # +batch dim

    self.features = {}                  # ⚠️ reset buffer trước mỗi ảnh
    with torch.no_grad():
        _ = self.yolo.model(tensor)     # hook tự chộp 3 layer
```

**Đọc code này thế nào**
- `permute(2, 0, 1)` đổi shape từ `(H, W, C)` (numpy/cv2) sang `(C, H, W)` (PyTorch convention). Quên bước này → model nhận shape sai, lỗi runtime.
- `/ 255.0` đưa pixel về 0-1. Quên → input cao gấp 255× lần model học → vector ra vô nghĩa (nhưng KHÔNG báo lỗi — đây là silent bug nguy hiểm).
- `self.features = {}` **bắt buộc reset** trước mỗi ảnh — nếu không, buffer còn feature ảnh cũ → vector ảnh mới bị nhiễm.
- `torch.no_grad()` tắt việc lưu graph cho gradient → giảm 30-50% RAM.
- Sau dòng `_ = self.yolo.model(tensor)`, **hook đã tự động** đẩy output 3 layer vào `self.features` — mình không cần gọi gì thêm. Đây là điểm hay của hook.

---

### 3.4. Pool 3 feature map xuống vector 512

**Concept**

Mình không cần biết "vết bệnh ở góc nào", chỉ cần "ảnh này có vết bệnh không". → Lấy **trung bình** cả tấm bản đồ thành **1 số duy nhất**.

```
Tấm 40×40 với 1600 ô  ─AvgPool 1×1─►  1 số (= trung bình 1600 ô)
Áp dụng cho 64 channels       →  64 số

Layer 4: [ 64 × 40 × 40]   → pool →  64 số
Layer 6: [128 × 20 × 20]   → pool → 128 số
Layer 9: [320 × 10 × 10]   → pool → 320 số
                              ─────────
                             Concat = 512 số  ✅
```

(Chú ý: YOLO11n có channels [64, 128, 320] = 512, không phải 64+128+256=448 như đoán ngây thơ.)

**Code** (`extract_deep_features.py:116-125`)
```python
    vectors = []
    for key in sorted(self.features.keys()):    # luôn sort: layer_4 → layer_6 → layer_9
        feat = self.features[key]
        pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        vectors.append(pooled.flatten().cpu().numpy())

    return np.concatenate(vectors)               # vector 512
```

**Đọc code này thế nào**
- `sorted(self.features.keys())` đảm bảo thứ tự **ổn định** (`layer_4, layer_6, layer_9`). Không sort → dict order phụ thuộc thứ tự đăng ký hook → giữa 2 lần chạy có thể đảo → ML học sai. Đây là lỗi cực kỳ khó debug nếu không cẩn thận.
- `adaptive_avg_pool2d(feat, (1, 1))` rất tiện: tự tính ratio để output đúng `1×1` bất kể input là 40×40, 20×20 hay 10×10. Không cần tính kernel/stride thủ công.
- `.cpu().numpy()` chuyển khỏi GPU → ghi CSV chạy trên CPU.
- `np.concatenate(vectors)` nối 3 vector con thành 1 vector dài 512.

---

### 3.5. Đọc nhãn dominant cho 1 ảnh (format YOLO)

**Concept**

YOLO label là **detection** (1 ảnh có thể có nhiều object, nhiều class). Nhưng classification cần **1 nhãn / ảnh**. Cách giải quyết: lấy **class xuất hiện NHIỀU NHẤT** trong các annotation của ảnh.

```
Một file label .txt có 5 dòng (mỗi dòng 1 box):
  0  ...   ← sau-an
  0  ...   ← sau-an
  0  ...   ← sau-an
  1  ...   ← benh-chay-la
  3  ...   ← benh-dom-mat-cua

class_ids = [0, 0, 0, 1, 3]
dominant_class = max(set(class_ids), key=class_ids.count)  # → 0 (sau-an)
```

→ Ảnh này gán nhãn `sau-an`.

**Code** (`extract_deep_features.py:228-240`)
```python
for img_path in tqdm(image_paths, desc=f"Extracting {split_key}"):
    label_path = label_dir / f"{img_path.stem}.txt"
    if not label_path.exists():
        continue                                # bỏ ảnh không có label

    with open(label_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if not labels:
        continue                                # label rỗng

    class_ids = [int(line.split()[0]) for line in labels]
    dominant_class = max(set(class_ids), key=class_ids.count)
```

**Đọc code này thế nào**
- File label YOLO format có dòng kiểu `0 0.5 0.5 0.3 0.4` — cột đầu là class id, các cột sau là tọa độ. `line.split()[0]` lấy đúng cột đầu.
- `max(set(...), key=class_ids.count)` là **idiom Python** để tìm phần tử xuất hiện nhiều nhất: `set()` để duyệt unique, `key=class_ids.count` đếm tần suất trên list gốc. Có thể thay bằng `Counter(class_ids).most_common(1)[0][0]` cũng được.
- 2 lần `continue` skip ảnh hỏng (không có label hoặc label rỗng) — bình thường có ~44/2400 ảnh bị skip, không đáng lo.

---

### 3.6. Loop qua hết 1 split → CSV

**Concept**

Hàm `extract_split` ghép tất cả: với mỗi ảnh trong train (hoặc valid/test) → đọc nhãn dominant → forward qua `YoloFeatureExtractor` → ghi 1 dòng vào DataFrame → cuối cùng `to_csv`.

**Code** (`extract_deep_features.py:202-264`, đoạn quan trọng)
```python
def extract_split(extractor, data_yaml_path, split_key, output_name, output_dir, max_images=None):
    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f)
    class_names = read_names(data_cfg)
    image_dir = resolve_split_dir(data_yaml_path, data_cfg[split_key])
    label_dir = image_dir.parent / "labels"
    image_paths = collect_image_paths(image_dir)

    rows = []
    for img_path in tqdm(image_paths, desc=f"Extracting {split_key}"):
        # ... (đọc dominant_class — xem 3.5)
        feat = extractor.extract_one(img_path)            # vector 512
        row = {f"feature_{i}": float(v) for i, v in enumerate(feat)}
        row["label"] = dominant_class
        row["class_name"] = class_names[dominant_class]
        row["image_name"] = img_path.name
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = output_dir / f"{output_name}_features.csv"
    df.to_csv(out_csv, index=False)
```

**Đọc code này thế nào**
- `read_names(data_cfg)` (lines 136-142) đọc class names từ `data.yaml`. Hỗ trợ cả format list (`["sau-an", "benh-chay-la", ...]`) lẫn dict (`{0: "sau-an", 1: "benh-chay-la", ...}`) — nên xử lý cả 2.
- `image_dir.parent / "labels"` là **convention YOLO**: ảnh ở `images/`, label ở `labels/` cùng cấp, cùng tên file (`IMG_001.jpg` ↔ `IMG_001.txt`).
- 1 row trong DataFrame có **515 cột**: `feature_0..feature_511` (512 cột) + `label` (int) + `class_name` (str) + `image_name` (str).
- `tqdm` chỉ để hiển thị progress bar — không ảnh hưởng logic.
- `extractor.extract_one(img_path)` (lines 127-133) là wrapper đọc file ảnh bằng cv2 → BGR → RGB → gọi `extract_array`.

---

### 3.7. Hai đường: YOLO format vs TensorFlow format

**Concept**

Dataset từ Roboflow có thể export ở 2 format:

| Format | Nhãn ở đâu | Khi nào dùng |
|---|---|---|
| **YOLO** | `labels/IMG_001.txt` (1 file/ảnh, mỗi dòng 1 box) | Có `data.yaml` chuẩn YOLO |
| **TensorFlow** | `train/_annotations.csv` (1 CSV/split, nhiều dòng/ảnh) | Khi không có `data.yaml` |

Script tự nhận: nếu thấy `_annotations.csv` thì TF, có `data.yaml` thì YOLO.

**Code** đáng chú ý (`extract_deep_features.py:171-199`)
```python
def load_tf_split_labels(split_dir):
    df = pd.read_csv(split_dir / "_annotations.csv")
    # ... validate cột "filename" + "class"

    label_by_image = {}
    for image_name, group in df.groupby("filename", sort=False):
        counts = group["class"].value_counts()
        dominant_class = str(counts.index[0])           # cũng dominant như YOLO format
        label_by_image[image_name] = dominant_class
    return label_by_image, class_order
```

**Đọc code này thế nào**
- TF format không có sẵn class id — chỉ có tên `"sau-an"`. Cần `build_tf_class_mapping` (lines 316-334) để gán số ổn định: `sorted(class_names)` → `{tên: idx}`. `sorted` quan trọng để 2 lần chạy ra cùng id.
- `groupby("filename")` gom các box cùng ảnh, rồi vẫn lấy **dominant** y hệt nhánh YOLO → cùng triết lý "1 nhãn / ảnh".
- `extract_split_tensorflow` (lines 267-313) cấu trúc gần giống `extract_split`, chỉ khác cách đọc nhãn. Code lặp lại 1 phần — chấp nhận được vì 2 format khá khác nhau.

---

### 3.8. Lắp ráp tất cả: `parse_args` + `main`

**Concept**

`main` là dây chuyền: parse CLI → check file tồn tại → tự nhận format → tạo `YoloFeatureExtractor` **1 lần** (load model nặng, không lặp) → gọi `extract_split` 3 lần cho train/valid/test.

**Code** (`extract_deep_features.py:361-440`, đoạn quan trọng)
```python
def main():
    args = parse_args()

    dataset_format = args.dataset_format
    if dataset_format == "auto":
        if (dataset_root / "train" / "_annotations.csv").exists() or ...:
            dataset_format = "tensorflow"
        elif data_yaml.exists():
            dataset_format = "yolo"
        else:
            raise FileNotFoundError(...)

    extractor = YoloFeatureExtractor(model_path, layer_indices, input_size=args.imgsz)

    if dataset_format == "yolo":
        extract_split(extractor, data_yaml, "train", "train", args.output_dir, args.max_images)
        extract_split(extractor, data_yaml, "val",   "valid", args.output_dir, args.max_images)
        extract_split(extractor, data_yaml, "test",  "test",  args.output_dir, args.max_images)
    else:
        class_to_id = build_tf_class_mapping(dataset_root)
        extract_split_tensorflow(extractor, dataset_root, "train", "train", ...)
        extract_split_tensorflow(extractor, dataset_root, "valid", "valid", ...)
        extract_split_tensorflow(extractor, dataset_root, "test",  "test",  ...)
```

**Đọc code này thế nào**
- `extractor = YoloFeatureExtractor(...)` chỉ tạo **1 lần** — load model best.pt + di chuyển lên GPU tốn vài giây, gọi 3 lần thì lãng phí.
- `--dataset-format auto`: nếu chỉ định `--dataset-format yolo` hoặc `tensorflow` thì skip auto-detect.
- Default từ `v2/config.py` (đã setup sẵn):
  - `DEFAULT_WEIGHTS = output/runs/detect/durian-1-yolo11n-smoke/weights/best.pt`
  - `DEFAULT_DATA_YAML = data/durian_refactor/data.yaml`
  - `DEFAULT_OUTPUT_DIR = output/extracted_features/`
- `--max-images N` để smoke-test nhanh (vd 50 ảnh thay vì 2400). Production thì bỏ flag này.

---

## 4. Tại sao chọn layer **4, 6, 9**?

| Layer | Thông tin | Nên lấy? |
|---|---|---|
| 0-3 | Cạnh, màu thô | ❌ Quá thấp, mọi ảnh giống nhau |
| **4-9** | **Texture, hình nhỏ** | ✅ Phong phú, phân biệt được bệnh |
| 10+ | Object hoàn chỉnh | ❌ Quá chuyên biệt cho detection task |

Lấy 3 layer (không phải 1) để có **đa độ phân giải** — vừa thấy chi tiết nhỏ vừa thấy hình tổng quan.

Muốn thử layer khác? Đổi `--layers 4,6,9` → ví dụ `--layers 5,7,9` rồi train lại Bước 5 và so accuracy.

---

## 5. ⚠️ Lưu ý quan trọng: `imgsz` phải KHỚP với train

Bước 3 train YOLO ở `imgsz=320` → Bước 4 extract cũng phải **320**.

Vì sao? Vì model học cách "nhìn" ảnh ở size đó. Đưa size khác (vd 640) vào → activation các layer sẽ khác → vector không còn "đúng nghĩa" với những gì model đã học.

```bash
uv run 04_features/extract_deep_features.py --imgsz 320   # ✅ ĐÚNG
uv run 04_features/extract_deep_features.py --imgsz 640   # ❌ SAI (mặc định argparse)
```

> Tip: nhớ check giá trị `imgsz` trong file `args.yaml` của run YOLO ở Bước 3 trước khi extract.

---

## 6. Output (`v2/output/extracted_features/`)

3 file CSV (mỗi split 1 file):

| File | Số ảnh | Số cột |
|---|---:|---:|
| `train_features.csv` | 2359 | 515 |
| `valid_features.csv` |  227 | 515 |
| `test_features.csv`  |  112 | 515 |

**515 cột = 512 features + label + class_name + image_name**

Mỗi dòng:
```
feature_0, feature_1, ..., feature_511, label, class_name,         image_name
0.12,      -0.04,     ..., -0.5,        0,     sau-an,             IMG_001.jpg
```

44/2403 ảnh train bị skip (file label rỗng hoặc không tồn tại) — bình thường, in `[WARN]` ở stdout.

---

## 7. 🎯 Phát hiện thú vị: Imbalance giảm mạnh

Vì giờ phân loại theo **ảnh** (1 nhãn/ảnh) thay vì theo **annotation** (nhiều/ảnh), class lớn (chay-la, sau-an) bị "gộp":

| Class | EDA (annotations) | Features (images) | Tỉ lệ giảm |
|---|---:|---:|---:|
| benh-chay-la | 4039 | 778 | ÷5.2 |
| sau-an | 3399 | 639 | ÷5.3 |
| benh-than-thu | 596 | 315 | ÷1.9 |
| benh-dom-mat-cua | 1113 | 222 | ÷5.0 |
| others | 605 | 204 | ÷3.0 |
| la-khoe-binh-thuong | 228 | 201 | ÷1.1 |

| | Imbalance ratio | Tác động |
|---|---:|---|
| EDA (annotations) | 18× | YOLO khó học class hiếm |
| Features (images) | **3.9×** | XGBoost dễ học hơn nhiều |

→ Bài toán classification ở Bước 5 sẽ **DỄ hơn** YOLO detection.

---

## 8. Lệnh chạy

```bash
cd v2
uv run 04_features/extract_deep_features.py --imgsz 320
```

Tham số khác (không bắt buộc):
- `--model <path>` — đổi model best.pt khác
- `--data-yaml <path>` — đổi dataset
- `--dataset-format auto|yolo|tensorflow` — fix nếu auto-detect đoán sai
- `--layers 4,6,9` — đổi layer nào lấy feature
- `--max-images N` — limit để test nhanh
- `--output-dir <path>` — đổi nơi xuất CSV

Thời gian: ~10-15 phút cho 2700 ảnh trên GTX 1650 (chủ yếu I/O đọc ảnh + forward CPU/GPU).

---

## 9. Sau Bước 4 thì đi đâu?

→ **Bước 5**: dùng 3 file CSV này train **XGBoost classifier**.
- Input: 512 cột feature
- Output: dự đoán 1 trong 6 class
- Đánh giá: accuracy, F1 trên `test_features.csv`

File: `v2/05_train_classifier/train_ml_features.py` — xem `v2/05_train_classifier/NOTES.md` cho chi tiết.

---

## 10. (Tùy chọn) `filter_top_classes.py` — lọc top-K class

### Mục đích
Sau khi train ML lần đầu (Bước 5), nếu có class quá yếu (F1 thấp / data quá ít), có thể **vứt bớt** rồi train lại để model "tập trung" vào class còn lại.

### Khi nào CẦN chạy?
| Tình huống | Có cần? |
|---|---|
| Dataset nhiều class (vd 15+) và 1 vài class rất yếu | ✅ Nên |
| Đã refactor 15→6 ở Bước 1 và muốn giữ cả 6 | ❌ KHÔNG cần |
| Muốn thử model "chỉ 4 class mạnh nhất" để demo | ✅ OK |

→ **Hiện tại không cần** vì đã refactor xuống 6 class ở Bước 1.

### Cách hoạt động
```
Input:
  v2/output/extracted_features/{train,valid,test}_features.csv  (6 class)
  v2/output/ml_models/classification_report_valid.csv           (F1 mỗi class)

  ↓ Rank class theo (valid_F1 desc, train_count desc)
  ↓ Bỏ class có train_count < 30 (data quá ít)
  ↓ Giữ top-K (default 6)
  ↓ Remap label cũ → label mới liên tục (0..K-1)

Output:
  v2/output/extracted_features_top6/{train,valid,test}_features.csv
  v2/output/extracted_features_top6/selection_summary.json
```

### 2 chế độ

**Chế độ AUTO** (mặc định) — rank theo F1:
```bash
uv run 04_features/filter_top_classes.py --top-k 4
```
→ Giữ 4 class có F1 cao nhất + đủ data.

**Chế độ MANUAL** — tự chỉ định:
```bash
uv run 04_features/filter_top_classes.py \
    --class-list "sau-an,benh-chay-la,la-khoe-binh-thuong,benh-dom-mat-cua"
```

### Lưu ý "remap label"
Sau khi bỏ class, label cũ có thể `[0, 1, 3, 4]` (thiếu 2). XGBoost cần label **liên tục** `[0, 1, 2, 3]`. Script tự đổi:
```
label cũ:  0  1  3  4
label mới: 0  1  2  3
```
Cột `label_raw` giữ giá trị gốc để debug.

### Pipeline nếu dùng filter
```
Bước 4a: extract_deep_features.py    → 6-class CSV
Bước 5a: train_ml_features.py        → ra report
Bước 4b: filter_top_classes.py       → 4-class CSV (nếu muốn lọc)
Bước 5b: train_ml_features.py \
            --features-dir output/extracted_features_top6 \
            --output-dir output/ml_models_top4
```

→ Train **2 lần**: lần 1 để biết class nào yếu, lần 2 sau khi lọc.

---

## Phụ lục A — Tham số chi tiết của extractor

> Bước 4 không "train" model nào, nên "hyperparameter" ở đây là các **lựa chọn cấu hình** ảnh hưởng đến vector output. Tổng cộng 6 tham số đáng biết.

### A.1. `input_size` (= `--imgsz`)
- **Default code**: `640` trong constructor, nhưng project mình dùng `320` (khớp với train Bước 3).
- **Ảnh hưởng**: output spatial size của 3 feature map.

| imgsz | Layer 4 (40×40 ở 320) | Layer 6 | Layer 9 | Tốc độ | RAM/ảnh |
|---:|---|---|---|---|---|
| 320 | 40×40 | 20×20 | 10×10 | nhanh ✅ | ~50MB |
| 416 | 52×52 | 26×26 | 13×13 | x1.7 chậm | ~80MB |
| 640 | 80×80 | 40×40 | 20×20 | x4 chậm | ~200MB |

> Vector cuối **luôn 512 chiều** dù imgsz nào (vì pool về 1×1) — chỉ tốc độ + chất lượng activation thay đổi.

### A.2. `layer_indices` (= `--layers`)
- **Default**: `[4, 6, 9]`.
- YOLO11n có 23 layer (0-22). Phân vùng:

| Vùng layer | Học gì | Recommend? |
|---|---|---|
| 0-3 | Cạnh, màu thô (pixel-level) | ❌ Quá thấp |
| **4-9** | **Texture, hình nhỏ (đốm, vân lá)** | ✅ Sweet spot |
| 10-15 | Object parts (cuống, mép lá) | ⚠️ Có thể thử |
| 16-22 | Detection head (chuyên cho box) | ❌ Quá đặc thù |

**Tradeoff số lượng layer chọn**:
- 1 layer (vd `[6]`): vector 128 chiều — ít info, có thể đủ cho task đơn giản.
- 3 layer (default): 512 chiều — đa scale, balance.
- 5+ layer (vd `[3,5,7,9,11]`): 700-1000+ chiều — dễ overfit ở Bước 5 với data nhỏ.

**Combo gợi ý cho thử nghiệm**:
- `4,6,9` (default): texture + middle features.
- `5,7,9`: bias high-level (vật thể hơn texture).
- `2,4,6`: bias low-level (texture/màu thô — nếu task nhạy cảm với màu).
- `4,6,9,11`: thử thêm 1 layer trên — vector 832 chiều.

### A.3. `device` (auto)
- Code: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- Không có CLI flag override — fix bằng env `CUDA_VISIBLE_DEVICES=""` nếu muốn ép CPU.
- GPU vs CPU: forward 1 ảnh 320×320 ~10ms (GPU) vs ~80ms (CPU). 2700 ảnh ≈ 27s vs 3.5 phút.

### A.4. Letterbox padding `(114, 114, 114)`
- **Magic number** của Ultralytics — YOLO được train với gray pad value này.
- Đổi sang `(0,0,0)` đen hay `(255,255,255)` trắng → activation các layer biên thay đổi → vector lệch khỏi phân bố train.
- ⚠️ **Không bao giờ thay đổi** nếu best.pt train với chuẩn Ultralytics.
- Nếu train YOLO custom với pad value khác → phải sửa cùng giá trị ở đây.

### A.5. Resize interpolation `cv2.INTER_LINEAR`
- Default trong `_letterbox`. Các option khác:

| Option | Tốc độ | Chất lượng | Use case |
|---|---|---|---|
| `INTER_NEAREST` | nhanh nhất | vỡ texture ❌ | KHÔNG dùng |
| **`INTER_LINEAR`** | nhanh | tốt ✅ | **Default** |
| `INTER_CUBIC` | chậm hơn 2× | mịn hơn | Overkill cho 320 |
| `INTER_AREA` | trung bình | tốt khi downsize lớn | OK nếu ảnh gốc ≥ 2000px |

Không cần đổi cho use case bệnh lá sầu riêng.

### A.6. Adaptive pool target `(1, 1)`
- Hiện pool về `1×1` → bỏ hết thông tin vị trí, chỉ giữ "có/không có pattern".
- Có thể đổi để giữ chút spatial info:

| Pool target | Output từ layer 4,6,9 | Tổng chiều | Ưu / Nhược |
|---|---|---:|---|
| **`(1,1)`** | 64+128+320 | **512** | Default. Đủ cho leaf disease. |
| `(2,2)` | (64+128+320)×4 | 2048 | Giữ vị trí thô (4 góc). Vector dài 4×. |
| `(4,4)` | (64+128+320)×16 | 8192 | Quá dài, dễ overfit. Cần data >5000. |

⚠️ **Đổi → phải retrain Bước 5** (vì vector dimension thay đổi).

### A.7. Tóm tắt: thường chỉ cần đổi 2 tham số
- `--imgsz` (khớp Bước 3): bắt buộc.
- `--layers` (nếu thử nghiệm khác): tùy chọn.
- 4 tham số còn lại (padding, interpolation, pool target, device) **để mặc định** trừ khi có lý do mạnh.

---

## Tóm tắt 1 dòng

> Bước 4 = bóp não YOLO ra **512 con số** cho mỗi ảnh, để Bước 5 (XGBoost) học classification. Trái tim file là class `YoloFeatureExtractor` với 5 method: `__init__`/`_register_hooks` (cắm camera) → `_letterbox` (chuẩn hóa ảnh) → `extract_array` (forward + pool) → `extract_one` (wrapper file). `filter_top_classes.py` là tùy chọn, chỉ dùng nếu muốn vứt bớt class yếu sau khi train xong lần 1.
