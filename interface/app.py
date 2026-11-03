
from pathlib import Path

import streamlit as st
from PIL import Image

from fish_freshness_demo import load_models, predict_image


# =========================
# CẤU HÌNH GIAO DIỆN
# =========================

PROJECT_DIR = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Fish Freshness Detection",
    page_icon="🐟",
    layout="wide"
)

st.title("🐟 HỆ THỐNG PHÂN LOẠI ĐỘ TƯƠI CỦA PHI LÊ CÁ")
st.markdown(
    "Ứng dụng xử lý ảnh sử dụng YOLO và MobileNetV3 "
    "để phát hiện và phân loại độ tươi của phi lê cá."
)

st.divider()


# =========================
# TẢI MÔ HÌNH
# =========================

@st.cache_resource
def get_models():
    return load_models()


# =========================
# GIAO DIỆN TẢI ẢNH
# =========================

st.subheader("1. Tải ảnh phi lê cá")

uploaded_file = st.file_uploader(
    "Chọn ảnh phi lê cá cần phân tích",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Ảnh đầu vào",
            use_container_width=True
        )

    with col2:
        st.subheader("Ảnh kết quả")
        st.info("Nhấn nút bên dưới để bắt đầu phân tích.")

    if st.button("🔍 PHÂN TÍCH ĐỘ TƯƠI", type="primary"):

        with st.spinner("Đang phân tích ảnh..."):

            try:
                detector, classifier = get_models()

                result_image = predict_image(
                    image,
                    detector,
                    classifier
                )

                st.success("Phân tích hoàn tất!")

                st.subheader("2. Kết quả nhận diện")

                st.image(
                    result_image,
                    caption="Ảnh sau khi phát hiện và phân loại",
                    use_container_width=True
                )

                # Hiển thị ảnh kết quả để tải xuống
                import io

                image_bytes = io.BytesIO()
                result_image.save(image_bytes, format="JPEG")

                st.download_button(
                    label="⬇️ Tải ảnh kết quả",
                    data=image_bytes.getvalue(),
                    file_name="fish_freshness_result.jpg",
                    mime="image/jpeg"
                )

            except Exception as error:
                st.error(f"Đã xảy ra lỗi: {error}")

else:
    st.info("Vui lòng tải ảnh phi lê cá lên để bắt đầu.")

st.divider()

st.caption(
    "Đề tài: Hệ thống phân loại độ tươi của phi lê cá trên băng chuyền "
    "| Môn học: Xử lý ảnh và Nhận dạng"
)
