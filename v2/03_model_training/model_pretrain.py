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

    model = YOLO("yolo11x.pt")

    results = model.train(
        data=str(DATA_YAML),
        project=str(RUNS_DIR),
        batch=8,
        epochs=100,
        device=0,
        imgsz=640,
        optimizer="AdamW",
        lr0=0.005,
        lrf=0.10,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        patience=50,
        seed=42,
        pretrained=True,
        save=True,
        val=True,
        plots=True,
        name=DEFAULT_RUN_NAME,
    )

    metrics = model.val(
        data=str(DATA_YAML),
        project=str(RUNS_DIR),
        name="val",
        imgsz=640,
        batch=8,
        save_json=True,
    )

    print(metrics)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # 👈 required on Windows
    multiprocessing.set_start_method('spawn', force=True)
    main()
