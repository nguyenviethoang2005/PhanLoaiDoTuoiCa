# HỆ THỐNG PHÂN LOẠI ĐỘ TƯƠI CỦA PHI LÊ CÁ TRÊN BĂNG CHUYỀN

## 1. Giới thiệu đề tài

Dự án xây dựng hệ thống ứng dụng xử lý ảnh và học sâu để phát hiện phi lê cá trong hình ảnh, sau đó phân loại độ tươi thành hai nhóm:

- **Fresh:** Phi lê cá tươi.
- **Spoiled:** Phi lê cá không tươi.

Hệ thống sử dụng YOLOv8 để phát hiện vị trí phi lê cá và MobileNetV3-Small để phân loại độ tươi của vùng ảnh được phát hiện.

Đây là mô hình thử nghiệm học thuật, chưa phải hệ thống kiểm định chất lượng hay an toàn thực phẩm trong thực tế.

## 2. Công nghệ và thư viện sử dụng

| Công nghệ | Mục đích |
|---|---|
| Python | Ngôn ngữ lập trình |
| OpenCV | Đọc và xử lý ảnh |
| Ultralytics YOLOv8 | Phát hiện phi lê cá |
| PyTorch | Xây dựng và huấn luyện mô hình |
| TorchVision | Cung cấp MobileNetV3-Small và xử lý ảnh |
| Scikit-learn | Đánh giá mô hình |
| Streamlit | Xây dựng giao diện demo |
| Pillow | Đọc và hiển thị ảnh |

## 3. Cấu trúc dự án

```text
Fish_Freshness_Detection/
├── classification/       # Huấn luyện và đánh giá phân loại
├── preprocessing/        # Tiền xử lý ảnh
├── interface/            # Giao diện ứng dụng
├── models/               # Trọng số mô hình
├── results/              # Hình ảnh và kết quả thực nghiệm
├── requirements.txt      # Danh sách thư viện
└── README.md             # Tài liệu hướng dẫn
```

## 4. Quy trình hoạt động

1. Nhận hình ảnh đầu vào.
2. Sử dụng YOLOv8 phát hiện vị trí phi lê cá.
3. Cắt vùng ảnh phi lê cá (ROI).
4. Đưa vùng ảnh vào MobileNetV3-Small.
5. Phân loại thành Fresh hoặc Spoiled.
6. Hiển thị kết quả và độ tin cậy trên giao diện.

## 5. Tiền xử lý ảnh

Dự án thực hiện các phương pháp:

- Resize ảnh về kích thước phù hợp.
- Giữ nguyên tỷ lệ ảnh bằng Letterbox khi chuẩn bị dữ liệu YOLO.
- Thử nghiệm CLAHE trên kênh độ sáng L của không gian màu LAB.
- Chuẩn hóa ảnh trước khi đưa vào mô hình phân loại.

CLAHE được thử nghiệm để so sánh chất lượng ảnh; mô hình phân loại cuối cùng sử dụng bộ ảnh Resize thông thường.

## 6. Cài đặt môi trường

Yêu cầu Python và các thư viện được liệt kê trong `requirements.txt`.

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

## 7. Chạy chương trình

Mở Terminal tại thư mục gốc dự án và chạy:

```bash
python -m streamlit run interface/app.py
```

Sau đó mở địa chỉ Local URL được Streamlit hiển thị trên Terminal.

**Lưu ý:** Cần có các file trọng số mô hình trong thư mục `models/` và đường dẫn trong mã nguồn phải chính xác.

## 8. Mô hình sử dụng

### YOLOv8
Dùng để phát hiện vị trí phi lê cá trong ảnh và tạo bounding box.

### MobileNetV3-Small
Dùng để phân loại vùng ảnh phi lê cá thành Fresh hoặc Spoiled. Mô hình được sử dụng thông qua PyTorch và TorchVision.

## 9. Đánh giá kết quả

Các tiêu chí đánh giá gồm:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- mAP và IoU đối với mô hình phát hiện YOLO

Kết quả thực nghiệm cho thấy mô hình còn hạn chế trong việc tổng quát hóa trên tập kiểm tra, đặc biệt có hiện tượng nhận nhầm ảnh Fresh thành Spoiled.

## 10. Hạn chế

- Dữ liệu chủ yếu là phi lê cá hồi vân.
- Nhãn Fresh/Spoiled được xây dựng theo tình trạng TVC trung bình từng ngày, không phải nhãn vi sinh riêng cho từng phi lê được chụp ảnh.
- Số lượng ảnh và nhóm cá còn hạn chế.
- Mô hình có thể phát hiện nhầm các vùng không phải phi lê cá.
- Hệ thống hiện tại là demo trên hình ảnh, chưa tích hợp camera, băng chuyền và cơ cấu điều khiển thực tế.

## 11. Hướng phát triển

- Thu thập thêm dữ liệu từ nhiều loại cá.
- Cải thiện chất lượng nhãn và số lượng mẫu.
- Nâng cao khả năng phát hiện phi lê cá.
- Cải thiện khả năng phân loại Fresh/Spoiled.
- Tích hợp camera và thử nghiệm trên băng chuyền thực tế.

## 12. Nguồn dữ liệu

Rainbow Trout Fillet Freshness Dataset:

https://zenodo.org/records/23600663



