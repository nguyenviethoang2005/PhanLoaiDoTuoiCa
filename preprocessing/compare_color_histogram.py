
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT = PROJECT_DIR / "dataset/processed/Rainbow_Trout"

DAY = 1

dir_a = ROOT / f"day{DAY}"
dir_b = ROOT / f"day{DAY}_clahe"

# Tìm ảnh có cùng tên trong hai bộ dữ liệu
files_a = {p.name for p in dir_a.glob("*M.jpg")}
files_b = {p.name for p in dir_b.glob("*M.jpg")}

common_files = sorted(files_a & files_b)

if not common_files:
    raise FileNotFoundError("Khong tim thay anh chung!")

filename = common_files[0]

image_a = cv2.imread(str(dir_a / filename))
image_b = cv2.imread(str(dir_b / filename))

if image_a is None or image_b is None:
    raise ValueError("Khong doc duoc anh!")

# Chuyển ảnh sang HSV
hsv_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2HSV)
hsv_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2HSV)

# Tính histogram của kênh Hue và Saturation
hist_h_a = cv2.calcHist([hsv_a], [0], None, [180], [0, 180])
hist_h_b = cv2.calcHist([hsv_b], [0], None, [180], [0, 180])

hist_s_a = cv2.calcHist([hsv_a], [1], None, [256], [0, 256])
hist_s_b = cv2.calcHist([hsv_b], [1], None, [256], [0, 256])

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(hist_h_a, label="A - Resize + Letterbox")
plt.plot(hist_h_b, label="B - Co CLAHE")
plt.title("Histogram kenh Hue")
plt.xlabel("Gia tri Hue")
plt.ylabel("So luong pixel")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(hist_s_a, label="A - Resize + Letterbox")
plt.plot(hist_s_b, label="B - Co CLAHE")
plt.title("Histogram kenh Saturation")
plt.xlabel("Gia tri Saturation")
plt.ylabel("So luong pixel")
plt.legend()
plt.grid()

plt.suptitle(f"So sanh mau sac: {filename}")
plt.tight_layout()
plt.show()
