
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

    canvas = cv2.copyMakeBorder(
        resized,
        (size - new_h) // 2,
        size - new_h - (size - new_h) // 2,
        (size - new_w) // 2,
        size - new_w - (size - new_w) // 2,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    return canvas

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(l_channel)

    result = cv2.merge((l_channel, a_channel, b_channel))

    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

total = 0

for day in range(1, 8):
    input_dir = INPUT_ROOT / f"day{day}" / "jpg"

    output_dir = OUTPUT_ROOT / f"day{day}_clahe"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Khong tim thay: {input_dir}")
        continue

    images = list(input_dir.glob("*M.jpg"))

    count = 0

    for image_path in images:
        image = cv2.imread(str(image_path))

        if image is None:
            print("Loi doc anh:", image_path.name)
            continue

        resized = resize_letterbox(image, TARGET_SIZE)
        processed = apply_clahe(resized)

        output_path = output_dir / image_path.name

        cv2.imwrite(str(output_path), processed)

        count += 1
        total += 1

    print(f"Day {day}: {count}/{len(images)} anh")

print("-" * 40)
print("TONG SO ANH DA XU LY:", total)
print("HOAN THANH!")
