import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ==============================================================================
# CẤU HÌNH GIAO DIỆN WEB (ĐẸP, HIỆN ĐẠI)
# ==============================================================================
st.set_page_config(
    page_title="Hệ Thống Nhận Diện Khuôn Mặt",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Thêm CSS custom để giao diện nhìn "pro" hơn
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    .stButton>button { background-color: #2563EB; color: white; border-radius: 8px; width: 100%; }
    .stButton>button:hover { background-color: #1D4ED8; color: white; }
    .predict-box { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #2563EB; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# TẢI MÔ HÌNH VÀ THÔNG TIN CLASS
# ==============================================================================
IMG_HEIGHT, IMG_WIDTH = 180, 180


@st.cache_resource
def load_my_model():
    # Load mô hình đã lưu từ Colab
    model = tf.keras.models.load_model("face_recognition_model.h5")
    return model


try:
    model = load_my_model()
    # DANH SÁCH TÊN NGƯỜI DÙNG (Cập nhật đúng thứ tự class_names từ Colab của bạn)
    # Ví dụ: class_names = ['An', 'Bình', 'Cường',...]
    # Bạn có thể tự động hóa bằng cách lưu list class_names thành file text từ Colab và đọc ở đây
    CLASS_NAMES = [
        "Người dùng 1",
        "Người dùng 2",
    ]  # <-- THAY THẾ DANH SÁCH NÀY CHO ĐÚNG VỚI DATA CỦA BẠN
except Exception as e:
    st.error(f"Không thể tải mô hình. Hãy chắc chắn file 'face_recognition_model.h5' nằm cùng thư mục. Lỗi: {e}")
    st.stop()


# Hàm xử lý và dự đoán ảnh
def predict_face(image):
    # Resize về chuẩn 180x180
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Tạo batch (1, 180, 180, 3)

    # Dự đoán
    predictions = model.predict(img_array)

    # Do mô hình dùng activation='softmax' ở tầng Dense cuối cùng,
    # predictions[0] chính là mảng xác suất của các class.
    score = predictions[0]
    class_idx = np.argmax(score)
    confidence = score[class_idx] * 100

    return CLASS_NAMES[class_idx], confidence


# ==============================================================================
# GIAO DIỆN CHÍNH
# ==============================================================================
st.title("👤 Hệ Thống Nhận Diện Khuôn Mặt Thông Minh")
st.markdown("Ứng dụng nhận diện danh tính thời gian thực sử dụng MobileNetV2.")
st.hr()

# Chia làm 2 cột: Cột 1 nhập dữ liệu ảnh, Cột 2 hiển thị kết quả phân tích
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Dữ liệu đầu vào")
    source_radio = st.radio(
        "Chọn nguồn cấp ảnh:", ("Sử dụng Camera trực tiếp", "Tải ảnh lên từ máy (.jpg, .png)")
    )

    final_image = None

    if source_radio == "Sử dụng Camera trực tiếp":
        # Tính năng Camera của Streamlit
        cam_image = st.camera_input("Đưa khuôn mặt vào khung hình")
        if cam_image is not None:
            final_image = Image.open(cam_image)

    else:
        uploaded_file = st.file_uploader("Chọn một tệp ảnh...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            final_image = Image.open(uploaded_file)
            st.image(final_image, caption="Ảnh đã tải lên", use_container_width=True)

with col2:
    st.subheader("🔍 Kết quả phân tích")

    if final_image is not None:
        with st.spinner("Đang xử lý dữ liệu ảnh và nhận diện..."):
            # Chạy hàm dự đoán
            label, confidence = predict_face(final_image)

        # Hiển thị UI kết quả cực đẹp
        st.markdown(
            f"""
            <div class="predict-box">
                <h3 style='margin-top:0; color:#1E3A8A;'>KẾT QUẢ</h3>
                <p style='font-size: 18px;'>Đối tượng nhận diện: <strong style='color:#EF4444; font-size: 24px;'>{label}</strong></p>
                <p style='font-size: 16px;'>Độ tin cậy: <strong>{confidence:.2f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Thanh progress bar biểu thị độ tin cậy
        st.progress(int(confidence))
    else:
        st.info("Vui lòng chụp ảnh hoặc tải ảnh lên để hệ thống thực hiện nhận diện.")