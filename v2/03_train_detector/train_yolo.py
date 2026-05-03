"""Smoke test train YOLO ~15 phút trên GTX 1650.

Mục đích: chạy thông pipeline + có best.pt để bước feature extraction
chạy được. KHÔNG kỳ vọng accuracy cao.

Phương án C: yolo11n + imgsz=320 + epochs=8 + fraction=1.0
Sau khi hiểu pipeline, đổi sang yolo11s/m + imgsz=640 + epochs>=50 cho
run thật.
"""

import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_YAML, DEFAULT_RUN_NAME, RUNS_DIR


def main():
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"CUDA version: {torch.version.cuda}")

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(DATA_YAML),
        project=str(RUNS_DIR),
        name=DEFAULT_RUN_NAME,
        epochs=8,
        imgsz=320,
        batch=16,
        fraction=1.0,
        device=device,
        optimizer="AdamW",
        lr0=0.005,
        lrf=0.10,
        weight_decay=0.0005,
        warmup_epochs=1,
        cos_lr=True,
        patience=8,
        seed=42,
        pretrained=True,
        cache="ram",
        save=True,
        save_period=-1,
        val=True,
        plots=True,
    )

    metrics = model.val(
        data=str(DATA_YAML),
        project=str(RUNS_DIR),
        name=f"{DEFAULT_RUN_NAME}-val",
        imgsz=320,
        batch=16,
        save_json=True,
    )

    print(metrics)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)
    main()
