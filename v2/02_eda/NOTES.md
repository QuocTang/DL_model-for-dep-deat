# 02 — EDA (Exploratory Data Analysis)

Mục đích: **NHÌN BẰNG MẮT** xem data trông thế nào trước khi train.

## File

### `plot_class_distribution.py`

Đọc tất cả nhãn ở `data/durian_refactor/{train,valid,test}/labels/`, đếm số annotation theo class, vẽ biểu đồ cột so sánh 3 split.

#### Đọc class names động từ `data.yaml`
Không hardcode → đổi data.yaml là plot tự cập nhật.

#### Backend headless
Trong WSL/SSH không có DISPLAY:
```python
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
```
→ `plt.savefig()` vẫn chạy, chỉ skip `plt.show()`.

## Output (`v2/output/eda/`)

| File | Mô tả |
|---|---|
| `class_distribution.png` | Biểu đồ cột 6 class × 3 split |
| `class_distribution.csv` | Bảng số gốc — copy vào báo cáo |

## Kết quả thực tế

| Class | train | valid | test | Total | % |
|---|---:|---:|---:|---:|---:|
| benh-chay-la | 4039 | 375 | 212 | 4626 | 41% |
| sau-an | 3399 | 316 | 154 | 3869 | 34% |
| benh-dom-mat-cua | 1113 | 132 | 46 | 1291 | 11% |
| benh-than-thu | 596 | 72 | 36 | 704 | 6% |
| others | 605 | 64 | 29 | 698 | 6% |
| **la-khoe-binh-thuong** | **228** | **15** | **10** | **253** | **2%** ⚠️ |

## Quan sát quan trọng

1. **Top 2 class chiếm ~75%** → model dễ "lười", chỉ học 2 class này nếu không xử
2. **`la-khoe-binh-thuong` chỉ 253 mẫu** — với 10 ảnh test, metric class này sẽ siêu nhiễu
3. **Imbalance ~18×** giữa class lớn nhất và nhỏ nhất
4. **Tỉ lệ split train:valid:test ≈ 89:8:3** (không phải 70:15:15 chuẩn) — do Roboflow auto-split

## Cách đọc biểu đồ

3 cột màu cho mỗi class: 🟦 train, 🟧 valid, 🟩 test.

Cần nhìn:
- Cột **train cao nhất**: bình thường
- **Bất kỳ class nào gần 0** ở valid/test: không thể đánh giá đáng tin cậy
- **Tỉ lệ giữa class**: imbalance cỡ nào → quyết định có cần `class_weights` không

## Lệnh chạy
```bash
cd v2
uv run 02_eda/plot_class_distribution.py
explorer.exe output/eda/class_distribution.png   # WSL → Windows viewer
```
