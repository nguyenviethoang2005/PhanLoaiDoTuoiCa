from ultralytics import YOLO
import torch

def main():
    # Kiem tra GPU
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("Khong tim thay GPU, su dung CPU")
        device = "cpu"

    # Tai mo hinh YOLOv8 Nano
    model = YOLO("yolov8n.pt")

    # Bat dau huan luyen
    results = model.train(
        data="dataset/yolo/fish.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=device,
        workers=2,
        patience=10,
        project="runs/fish_detection",
        name="yolov8n_fillet_adamw",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        plots=True,
        save=True
    )

    print("\nHUAN LUYEN HOAN TAT!")
    print("Ket qua:", results.save_dir)


if __name__ == "__main__":
    main()