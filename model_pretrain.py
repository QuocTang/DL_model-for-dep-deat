import torch
from ultralytics import YOLO

def main():
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"CUDA version: {torch.version.cuda}")

    model = YOLO("yolo11x.pt")

    results = model.train(
        data=r"C:\Users\ADMIN\Desktop\personal_train\ML_deppfeat\durian-1\data.yaml",
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
        name="durian-1-yolo11x"
    )

    metrics = model.val(
        data=r"C:\Users\ADMIN\Desktop\personal_train\ML_deppfeat\durian-1\data.yaml",
        imgsz=640,
        batch=8,
        save_json=True
    )

    print(metrics)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # 👈 required on Windows
    multiprocessing.set_start_method('spawn', force=True)
    main()
