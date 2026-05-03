import os
from pathlib import Path

# Đường dẫn gốc dữ liệu (source) và thư mục đích (destination) cho việc refactor.
# Sử dụng đường dẫn tương đối tới thư mục `data` trong dự án để tránh hard‑code đường dẫn Windows.
# Thay đổi nếu muốn lưu trữ dữ liệu ở vị trí khác.
SRC_ROOT = Path(__file__).resolve().parent / "data" / "durian-1"
DST_ROOT = Path(__file__).resolve().parent / "data" / "durian_refactor"

# ROOT_DIR được dùng trong một số script (vd. downsample_large_classes) để trỏ tới dữ liệu gốc.
# Nếu muốn dùng cùng một đường dẫn, gán ROOT_DIR = SRC_ROOT.
ROOT_DIR = SRC_ROOT

