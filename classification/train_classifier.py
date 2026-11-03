
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from pathlib import Path
import copy


# ==========================================
# 1. CAU HINH
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "dataset" / "classification"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.0001
PATIENCE = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Thiet bi:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 2. TIEN XU LY ANH
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
    brightness=0.05,
    contrast=0.05,
    saturation=0.0
),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. DOC DU LIEU
# ==========================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=val_test_transform
)

print("\nCac lop:", train_dataset.classes)
print("So anh Train:", len(train_dataset))
print("So anh Validation:", len(val_dataset))
print("So anh Test:", len(test_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 4. KHOI TAO MOBILENETV3-SMALL
# ==========================================

weights = models.MobileNet_V3_Small_Weights.DEFAULT

model = models.mobilenet_v3_small(weights=weights)

# Thay lop phan loai cu bang 2 lop Fresh va Spoiled
in_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(in_features, 2)

model = model.to(device)


# ==========================================
# 5. HAM MAT MAT VA TOI UU
# ==========================================

# Can bang trong so cho hai lop
class_counts = torch.bincount(
    torch.tensor(train_dataset.targets)
).float()

class_weights = torch.tensor(
    [1.0, 2.0],
    dtype=torch.float32
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device)
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 6. HUAN LUYEN
# ==========================================

best_accuracy = 0.0
best_model_weights = copy.deepcopy(model.state_dict())

epochs_without_improvement = 0

print("\n===== BAT DAU HUAN LUYEN =====")

for epoch in range(EPOCHS):

    # ----- TRAIN -----
    model.train()

    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()

    train_loss /= len(train_dataset)
    train_accuracy = train_correct / len(train_dataset)


    # ----- VALIDATION -----
    model.eval()

    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()

    val_loss /= len(val_dataset)
    val_accuracy = val_correct / len(val_dataset)


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )


    # ----- LUU MO HINH TOT NHAT -----
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": best_model_weights,
                "class_names": train_dataset.classes
            },
            MODEL_DIR / "best_classifier.pth"
        )

        epochs_without_improvement = 0

        print("  -> Da luu mo hinh tot nhat!")

    else:
        epochs_without_improvement += 1


    # ----- EARLY STOPPING -----
    if epochs_without_improvement >= PATIENCE:

        print("\nDung som do Validation khong cai thien.")
        break


# ==========================================
# 7. KET THUC
# ==========================================

model.load_state_dict(best_model_weights)

print("\n===== HUAN LUYEN HOAN TAT =====")
print("Validation Accuracy tot nhat:", round(best_accuracy, 4))
print("Mo hinh da luu tai:")
print(MODEL_DIR / "best_classifier.pth")
