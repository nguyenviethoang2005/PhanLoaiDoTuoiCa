
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Xác định thư mục gốc của dự án
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Tên ảnh mẫu
image_name = "20260509F1F10M.jpg"

# Đường dẫn đến 3 giai đoạn ảnh
image_paths = [
    PROJECT_DIR / "dataset/raw/Rainbow_Trout/day1/jpg" / image_name,
    PROJECT_DIR / "dataset/processed/Rainbow_Trout/day1" / image_name,
    PROJECT_DIR / "dataset/processed/Rainbow_Trout/day1_clahe" / image_name
]

titles = [
    "Original Image",
    "Resize + Letterbox",
    "Resize + Letterbox + CLAHE"
]

# Tạo khung hiển thị
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

for ax, path, title in zip(axes, image_paths, titles):
    image = cv2.imread(str(path))

    if image is None:
        print(f"KHONG TIM THAY ANH: {path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()

# Lưu kết quả
output_dir = PROJECT_DIR / "results"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "preprocessing_comparison.png"

plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Da luu anh tai: {output_path}")

plt.show()
