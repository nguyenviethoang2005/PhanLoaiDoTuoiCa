
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "dataset" / "classification"
MODEL_PATH = PROJECT_DIR / "models" / "best_classifier.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tien xu ly anh Validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

# Khoi tao MobileNetV3-Small
model = models.mobilenet_v3_small(weights=None)

in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, 2)

# Nap trong so tot nhat
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Danh sach ten lop
class_names = test_dataset.classes

# Du doan
all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        # Chuyen dau ra thanh xac suat
        probabilities = torch.softmax(outputs, dim=1)

        # Lay lop co xac suat cao nhat
        predictions = (probabilities[:, 1] >= 0.35).long()

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())

        # Luu xac suat thuoc lop Spoiled
        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )

# In xac suat du doan tung anh
print("\n===== XAC SUAT DU DOAN TUNG ANH =====")

for i, (image_path, true_label) in enumerate(test_dataset.samples):

    probability = all_probabilities[i]

    predicted_label = class_names[all_predictions[i]]

    print(
        f"{Path(image_path).name} | "
        f"Thuc te: {class_names[true_label]} | "
        f"Du doan: {predicted_label} | "
        f"Xac suat Spoiled: {probability:.4f}"
    )

# Danh gia mo hinh
print("\n===== KET QUA DANH GIA VALIDATION =====")

accuracy = accuracy_score(all_labels, all_predictions)
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))

print("\nDanh gia hoan tat!")
