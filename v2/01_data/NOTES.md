# 01 — Data acquisition

Bước đầu của pipeline: lấy data về máy + xử nhãn.

## Files

### `pull_data.py`
Tải dataset từ Roboflow (`cico-siefo/durian-k51j3`, version 1, format `yolov11`).
Output: `v2/data/durian-1/` với cấu trúc YOLO:
```
durian-1/
├── data.yaml      ← config: nc=15, names=[...]
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

> ⚠️ **Bảo mật**: file đang hardcode API key Roboflow. Để an toàn nên đổi sang env var: `os.getenv("ROBOFLOW_API_KEY")`.

### `refactor_data.py`
Remap **15 class → 6 class** để giảm độ phức tạp cho lần đầu train.

```python
KEEP_IDS = {
    12: 0,  # sau-an
    2:  1,  # benh-chay-la
    7:  2,  # benh-than-thu
    3:  3,  # benh-dom-mat-cua
    10: 4,  # la-khoe-binh-thuong
}
# Mọi class khác → 5 (others)
```

Đọc từng file `.txt` ở `durian-1/`, đổi `parts[0]` (class_id) sang ID mới, ghi sang `durian_refactor/`. Tọa độ giữ nguyên.

#### 🐛 Bug đã fix
Ban đầu script có check `if len(parts) != 5: continue` — assume label format **bbox** (5 token). Nhưng data thực tế là **YOLO segmentation polygon** (`class x1 y1 x2 y2 ... xn yn`, có thể 11/19/21/27+ token).

→ Script bỏ qua ~99% nhãn. Fix: đổi thành `< 5`.

```python
# Trước:
if len(parts) != 5:
    continue
print(parts)              # in spam vài ngàn dòng

# Sau:
if len(parts) < 5:        # cho phép cả bbox lẫn polygon
    continue
```

### `downsample_large_classes.py` — ĐÃ SKIP

Script tùy chọn để giảm 60% ảnh class lớn (`benh-chay-la`, `sau-an`) → cân bằng dataset. **Lần đầu không chạy** vì:
- Mất 60% data → có thể yếu hơn
- Chỉ xử lý split `train`, không đụng valid/test
- `keep_ratio = 0.4` arbitrary, hardcoded
- Output `durian-1_balanced/` không kết nối với refactor pipeline
- Bỏ ảnh có class lớn → vô tình bỏ luôn class hiếm trong cùng ảnh
- ⚠️ **Phải chạy TRƯỚC `refactor_data.py`** (dùng class ID gốc 2, 12)

→ Cách balance tốt hơn: dùng `class_weights` trong YOLO `model.train()` sau khi có baseline.

## Phân bố sau refactor

| New ID | Class | Annotations (train) |
|---:|---|---:|
| 0 | sau-an | 3399 |
| 1 | benh-chay-la | 4039 |
| 2 | benh-than-thu | 596 |
| 3 | benh-dom-mat-cua | 1113 |
| 4 | **la-khoe-binh-thuong** | **228** ⚠️ |
| 5 | others (gộp 10 class còn lại) | 605 |

Imbalance ~18× giữa class lớn nhất và nhỏ nhất.

## Lệnh chạy
```bash
cd v2
uv run 01_data/refactor_data.py    # đã làm
# uv run 01_data/downsample_large_classes.py   # SKIP
```
