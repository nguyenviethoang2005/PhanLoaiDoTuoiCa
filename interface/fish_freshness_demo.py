
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent

YOLO_PATH = (
    PROJECT_DIR
    / "runs/detect/runs/fish_detection/"
    / "yolov8n_fillet/weights/best.pt"
)

CLASSIFIER_PATH = PROJECT_DIR / "models" / "best_classifier.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Fresh", "Spoiled"]
SPOILED_THRESHOLD = 0.35

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_models():
    print("Đang tải mô hình...")

    detector = YOLO(str(YOLO_PATH))

    classifier = models.mobilenet_v3_small(weights=None)
    in_features = classifier.classifier[-1].in_features
    classifier.classifier[-1] = nn.Linear(in_features, 2)

    checkpoint = torch.load(
        CLASSIFIER_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    classifier.load_state_dict(checkpoint)
    classifier.to(DEVICE)
    classifier.eval()

    print("Đã tải xong hai mô hình!")
    print("Thiết bị:", DEVICE)

    return detector, classifier


def classify_fish(crop, classifier):
    image_tensor = transform(crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = classifier(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    spoiled_probability = probabilities[1].item()

    predicted_class = (
        1 if spoiled_probability >= SPOILED_THRESHOLD else 0
    )

    confidence = probabilities[predicted_class].item()

    return CLASS_NAMES[predicted_class], confidence


def predict_image(image_path, detector, classifier):

    # Hỗ trợ cả ảnh PIL từ Streamlit và đường dẫn ảnh
    if isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    results = detector.predict(
        source=image,
        conf=0.50,
        iou=0.45,
        verbose=False
    )

    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    detections = results[0].boxes

    if detections is None or len(detections) == 0:
        print("Không phát hiện được phi lê cá.")
        return result_image

    for index, box in enumerate(detections):

        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

        # Độ tin cậy phát hiện của YOLO
        yolo_confidence = box.conf[0].item()

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(image.width, int(x2))
        y2 = min(image.height, int(y2))

        if x2 <= x1 or y2 <= y1:
            continue

        # Cắt vùng phi lê cá
        crop = image.crop((x1, y1, x2, y2))

        # Phân loại độ tươi
        label, confidence = classify_fish(crop, classifier)

        color = "green" if label == "Fresh" else "red"

        text = (
            f"{label}: {confidence:.2%} "
            f"| YOLO: {yolo_confidence:.2%}"
        )

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=3
        )

        draw.text(
            (x1, max(0, y1 - 20)),
            text,
            fill=color
        )

        print(
            f"Phi lê {index + 1}: {label} "
            f"- độ tin cậy phân loại: {confidence:.2%} "
            f"- độ tin cậy YOLO: {yolo_confidence:.2%}"
        )

    return result_image


if __name__ == "__main__":

    detector, classifier = load_models()

    IMAGE_PATH = (
        PROJECT_DIR
        / "dataset/processed/Rainbow_Trout/day1"
        / "20260509F1F1M.jpg"
    )

    if not IMAGE_PATH.exists():
        print("Không tìm thấy ảnh:", IMAGE_PATH)

    else:
        output = predict_image(
            IMAGE_PATH,
            detector,
            classifier
        )

        output_path = PROJECT_DIR / "results" / "integrated_demo.jpg"
        output.save(output_path)

        print("Đã lưu ảnh kết quả:", output_path)
