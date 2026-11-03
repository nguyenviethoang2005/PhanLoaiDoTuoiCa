
from pathlib import Path
import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_ROOT = PROJECT_DIR / "dataset/raw/Rainbow_Trout"
OUTPUT_ROOT = PROJECT_DIR / "dataset/processed/Rainbow_Trout"

TARGET_SIZE = 640


def resize_letterbox(image, size=640):
    h, w = image.shape[:2]

    scale = min(size / w, size / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left

    result = cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    return result


total = 0

for day in range(1, 8):
    input_dir = INPUT_ROOT / f"day{day}" / "jpg"

    output_dir = OUTPUT_ROOT / f"day{day}"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(input_dir.glob("*M.jpg"))

    count = 0

    for image_path in images:
        image = cv2.imread(str(image_path))

        if image is None:
            print("Loi doc:", image_path.name)
            continue

        processed = resize_letterbox(image, TARGET_SIZE)

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), processed)

        count += 1
        total += 1

    print(f"Day {day}: {count}/{len(images)} anh")

print("-" * 40)
print("TONG SO ANH DA XU LY:", total)
print("HOAN THANH!")
