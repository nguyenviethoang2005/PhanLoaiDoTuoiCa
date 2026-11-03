
from pathlib import Path
import cv2

# Thư mục chứa ảnh gốc
INPUT_DIR = Path("../dataset/raw/Rainbow_Trout/day1/jpg")

# Thư mục lưu ảnh sau tiền xử lý
OUTPUT_DIR = Path("../dataset/processed/Rainbow_Trout/day1")

# Kích thước chuẩn
TARGET_SIZE = 640

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lấy ảnh mặt thịt cá M.jpg
image_paths = sorted(INPUT_DIR.glob("*M.jpg"))

print("Tổng số ảnh:", len(image_paths))


def letterbox(image, size=640):
    """
    Resize ảnh giữ nguyên tỷ lệ,
    sau đó thêm viền đen để đạt kích thước vuông.
    """

    h, w = image.shape[:2]

    # Tính tỷ lệ resize
    scale = min(size / w, size / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize ảnh
    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Tạo nền vuông màu đen
    canvas = image.copy()
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


# Xử lý từng ảnh
for image_path in image_paths:

    image = cv2.imread(str(image_path))

    if image is None:
        print("Lỗi đọc ảnh:", image_path.name)
        continue

    processed = letterbox(image, TARGET_SIZE)

    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(str(output_path), processed)

    print("Đã xử lý:", image_path.name)

print("\nHoàn thành tiền xử lý Resize + Letterbox!")
