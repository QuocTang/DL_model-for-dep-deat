"""Generate professional pipeline diagram for Chapter 3."""
"""durian-leaf-disease-detection/.venv/bin/python docs/report/generate_pipeline.py"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor("#FAFBFD")

# ── Gradient-like background banner ──
banner = mpatches.FancyBboxPatch(
    (0.3, 12.4), 21.4, 1.3,
    boxstyle="round,pad=0.15", facecolor="#1A237E",
    edgecolor="none", alpha=0.95)
ax.add_patch(banner)
ax.text(11, 13.05,
        "SƠ ĐỒ QUY TRÌNH TỔNG THỂ XÂY DỰNG MÔ HÌNH PHÁT HIỆN BỆNH TRÊN LÁ SẦU RIÊNG",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="white", family="sans-serif")

# ── Phase config ──
phases = [
    {
        "title": "GIAI ĐOẠN 1",
        "subtitle": "Thu thập dữ liệu",
        "icon": "[1]",
        "color": "#2E7D32", "light": "#E8F5E9", "border": "#4CAF50",
        "items": [
            ("Nguồn", "Roboflow API"),
            ("Workspace", "cico-siefo"),
            ("Project", "durian-k51j3, v1"),
            ("Định dạng", "YOLOv11"),
            ("Cấu trúc", "images/ + labels/"),
            ("Phân chia", "train / valid / test"),
            ("Số lớp gốc", "15 lớp bệnh/trạng thái"),
        ],
    },
    {
        "title": "GIAI ĐOẠN 2",
        "subtitle": "Khảo sát & Phân tích (EDA)",
        "icon": "[2]",
        "color": "#1565C0", "light": "#E3F2FD", "border": "#42A5F5",
        "items": [
            ("Công cụ", "Seaborn, Matplotlib"),
            ("Biểu đồ", "Bar plot phân bố lớp"),
            ("Tập phân tích", "Train + Valid"),
            ("Phát hiện", "Mất cân bằng nghiêm trọng"),
            ("Lớp lớn", "Cháy lá (2), Sâu ăn (12)"),
            ("Kết luận", "Cần tiền xử lý cân bằng"),
        ],
    },
    {
        "title": "GIAI ĐOẠN 3",
        "subtitle": "Tiền xử lý dữ liệu",
        "icon": "[3]",
        "color": "#E65100", "light": "#FFF3E0", "border": "#FF9800",
        "items": [
            ("Bước 1", "Gom nhóm: 15 → 6 lớp"),
            ("  Lớp 0", "Sâu ăn (ID gốc: 12)"),
            ("  Lớp 1", "Bệnh cháy lá (ID gốc: 2)"),
            ("  Lớp 2", "Bệnh thán thư (ID gốc: 7)"),
            ("  Lớp 3", "Đốm mắt cua (ID gốc: 3)"),
            ("  Lớp 4", "Lá khỏe (ID gốc: 10)"),
            ("  Lớp 5", "Other (các lớp còn lại)"),
            ("Bước 2", "Downsample lớp 2,12 → 40%"),
            ("Bước 3", "Copy ảnh + viết lại label"),
        ],
    },
    {
        "title": "GIAI ĐOẠN 4",
        "subtitle": "Huấn luyện mô hình",
        "icon": "[4]",
        "color": "#6A1B9A", "light": "#F3E5F5", "border": "#AB47BC",
        "items": [
            ("Mô hình", "YOLOv11x (Extra-Large)"),
            ("Pretrained", "yolo11x.pt (COCO)"),
            ("Chiến lược", "Transfer Learning + Fine-tune"),
            ("Optimizer", "AdamW"),
            ("Learning rate", "lr₀=0.005, lrf=0.10"),
            ("LR Scheduler", "Cosine Annealing"),
            ("Epochs / Batch", "100 epochs, batch=8"),
            ("Image size", "640 × 640"),
            ("Early Stopping", "patience = 50"),
            ("Augmentation", "Mosaic, Flip, HSV, Erasing"),
            ("Seed", "42 (deterministic)"),
        ],
    },
    {
        "title": "GIAI ĐOẠN 5",
        "subtitle": "Đánh giá mô hình",
        "icon": "[5]",
        "color": "#C62828", "light": "#FFEBEE", "border": "#EF5350",
        "items": [
            ("Precision", "Tỷ lệ dự đoán đúng"),
            ("Recall", "Tỷ lệ phát hiện đúng"),
            ("mAP@0.5", "AP trung bình tại IoU=0.5"),
            ("mAP@0.5:0.95", "AP trung bình đa ngưỡng"),
            ("Confusion Matrix", "Ma trận nhầm lẫn"),
            ("Loss curves", "train/val box, cls, dfl"),
            ("PR Curve", "Precision-Recall curve"),
        ],
    },
]

# ── Layout: 5 columns ──
col_w = 3.8
col_gap = 0.35
x_start = 0.5
y_header = 11.6
y_body_top = 10.2

for i, p in enumerate(phases):
    cx = x_start + i * (col_w + col_gap)

    # ── Phase header card ──
    header_h = 1.6
    hdr = mpatches.FancyBboxPatch(
        (cx, y_header - header_h / 2), col_w, header_h,
        boxstyle="round,pad=0.12", facecolor=p["color"],
        edgecolor="white", linewidth=2.5, alpha=0.95,
        zorder=3)
    # Shadow
    shadow = mpatches.FancyBboxPatch(
        (cx + 0.06, y_header - header_h / 2 - 0.06), col_w, header_h,
        boxstyle="round,pad=0.12", facecolor="#00000022",
        edgecolor="none", zorder=2)
    ax.add_patch(shadow)
    ax.add_patch(hdr)

    # Phase number + icon
    ax.text(cx + col_w / 2, y_header + 0.28, p["icon"] + "  " + p["title"],
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="#FFFFFFCC", family="sans-serif", zorder=4)
    ax.text(cx + col_w / 2, y_header - 0.25, p["subtitle"],
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color="white", family="sans-serif", zorder=4)

    # ── Arrow to next phase ──
    if i < len(phases) - 1:
        ax.annotate("",
            xy=(cx + col_w + col_gap - 0.08, y_header),
            xytext=(cx + col_w + 0.08, y_header),
            arrowprops=dict(arrowstyle="-|>", color=p["color"],
                            lw=3, mutation_scale=20),
            zorder=5)

    # ── Vertical connector ──
    ax.plot([cx + col_w / 2, cx + col_w / 2],
            [y_header - header_h / 2, y_body_top + 0.1],
            color=p["border"], lw=2, ls="--", alpha=0.6, zorder=1)
    ax.annotate("",
        xy=(cx + col_w / 2, y_body_top + 0.05),
        xytext=(cx + col_w / 2, y_body_top + 0.4),
        arrowprops=dict(arrowstyle="-|>", color=p["border"], lw=1.8,
                        mutation_scale=14), zorder=3)

    # ── Detail card ──
    n = len(p["items"])
    row_h = 0.42
    card_h = n * row_h + 0.6
    card_y = y_body_top - card_h

    # Card shadow
    cs = mpatches.FancyBboxPatch(
        (cx + 0.05, card_y - 0.05), col_w, card_h,
        boxstyle="round,pad=0.1", facecolor="#00000015",
        edgecolor="none", zorder=1)
    ax.add_patch(cs)
    # Card body
    card = mpatches.FancyBboxPatch(
        (cx, card_y), col_w, card_h,
        boxstyle="round,pad=0.1", facecolor=p["light"],
        edgecolor=p["border"], linewidth=1.5, alpha=0.92, zorder=2)
    ax.add_patch(card)

    # Items
    for j, (key, val) in enumerate(p["items"]):
        ty = y_body_top - 0.45 - j * row_h
        is_indent = key.startswith("  ")
        k = key.strip()

        if is_indent:
            ax.text(cx + 0.35, ty, "◦", ha="left", va="center",
                    fontsize=8, color=p["color"], family="sans-serif", zorder=3)
            ax.text(cx + 0.55, ty, k, ha="left", va="center",
                    fontsize=8.2, fontweight="bold", color="#444",
                    family="sans-serif", zorder=3)
            ax.text(cx + 1.45, ty, val, ha="left", va="center",
                    fontsize=8.2, color="#555", family="sans-serif", zorder=3)
        else:
            ax.text(cx + 0.2, ty, "●", ha="left", va="center",
                    fontsize=5, color=p["color"], family="sans-serif", zorder=3)
            ax.text(cx + 0.42, ty, k + ":", ha="left", va="center",
                    fontsize=8.5, fontweight="bold", color="#333",
                    family="sans-serif", zorder=3)
            # Estimate text x position after key
            kx = cx + 0.42 + len(k) * 0.075 + 0.15
            ax.text(kx, ty, val, ha="left", va="center",
                    fontsize=8.5, color="#555", family="sans-serif", zorder=3)

# ── Bottom data flow bar ──
flow_y = 0.7
flow_bar = mpatches.FancyBboxPatch(
    (0.5, flow_y - 0.35), 21, 0.7,
    boxstyle="round,pad=0.1", facecolor="#ECEFF1",
    edgecolor="#B0BEC5", linewidth=1, alpha=0.8)
ax.add_patch(flow_bar)

flow_labels = [
    ("Dữ liệu thô\n(15 lớp, raw)", 2.5),
    ("→", 4.7),
    ("Dữ liệu phân tích\n(EDA insights)", 6.5),
    ("→", 8.7),
    ("Dữ liệu sạch\n(6 lớp, cân bằng)", 10.5),
    ("→", 12.7),
    ("Mô hình YOLOv11x\n(best weights)", 14.5),
    ("→", 16.7),
    ("Kết quả đánh giá\n(metrics + plots)", 18.5),
]
for label, fx in flow_labels:
    if label == "→":
        ax.text(fx, flow_y, "▶", ha="center", va="center",
                fontsize=12, color="#78909C", family="sans-serif")
    else:
        ax.text(fx, flow_y, label, ha="center", va="center",
                fontsize=8, color="#455A64", family="sans-serif",
                fontweight="bold")

plt.tight_layout(pad=0.5)
plt.savefig(
    "/Users/quoctang/workspaces/master/DL_model-for-dep-deat/docs/report/hinh_3_1_pipeline.png",
    dpi=220, bbox_inches="tight", facecolor="#FAFBFD",
)
print("Saved: hinh_3_1_pipeline.png")
