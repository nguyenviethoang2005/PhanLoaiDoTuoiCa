from ultralytics import YOLO

# Duong dan mo hinh tot nhat
model = YOLO(
    r"runs/detect/runs/fish_detection/yolov8n_fillet_adamw/weights/best.pt"
)

# Danh gia tren tap Test
metrics = model.val(
    data="dataset/yolo/fish.yaml",
    split="test",
    imgsz=640,
    device=0
)

print("\n===== KET QUA DANH GIA TEST =====")
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)