import os
import gdown
import numpy as np
import streamlit as st
import tflite_runtime.interpreter as tflite
from PIL import Image

# ==============================================================================
# CẤU HÌNH GIAO DIỆN WEB
# ==============================================================================
st.set_page_config(
    page_title="Hệ Thống Nhận Diện Khuôn Mặt",
    page_icon="👤",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    .predict-box { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #2563EB; }
    </style>
    """,
    unsafe_allow_html=True,
)

IMG_HEIGHT, IMG_WIDTH = 180, 180
MODEL_PATH = "face_recognition_model.tflite"

# ==============================================================================
# TỰ ĐỘNG TẢI MÔ HÌNH TFLITE TỪ GOOGLE DRIVE
# ==============================================================================
@st.cache_resource
def load_tflite_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Đang tải file mô hình siêu nhẹ từ Google Drive..."):
            # CHÚ Ý: Thay đoạn mã ID bên dưới bằng ID file .tflite mới của bạn trên Drive
            google_drive_id = "ĐIỀN_ID_FILE_TFLITE_MỚI_VÀO_ĐÂY" 
            url = f"https://drive.google.com/uc?id={google_drive_id}"
            try:
                gdown.download(url, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Không thể tải file mô hình. Lỗi: {e}")
                st.stop()
            
    try:
        # Khởi tạo bộ thông dịch TFLite thay cho Keras Model
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo mô hình TFLite: {e}")
        st.stop()

try:
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Cập nhật danh sách tên người dùng của bạn tại đây (phải đúng thứ tự lúc train)
    CLASS_NAMES = ["Người dùng 1", "Người dùng 2"] 
except Exception as e:
    st.error(f"❌ Không thể khởi tạo hệ thống: {e}")
    st.stop()

def predict_face(image):
    # Chuẩn hóa ảnh về kích thước yêu cầu và định dạng float32
    img = image.convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Tạo batch (1, 180, 180, 3)
    
    # Đưa dữ liệu ảnh vào mô hình TFLite
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    # Lấy kết quả dự đoán đầu ra
    predictions = interpreter.get_tensor(output_details[0]['index'])
    score = predictions[0]
    class_idx = np.argmax(score)
    confidence = float(score[class_idx]) * 100
    
    return CLASS_NAMES[class_idx], confidence

# ==============================================================================
# GIAO DIỆN CHÍNH
# ==============================================================================
st.title("👤 Hệ Thống Nhận Diện Khuôn Mặt Thông Minh")
st.markdown("Ứng dụng chạy trực tuyến sử dụng TFLite siêu tốc.")
st.hr()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Dữ liệu đầu vào")
    source_radio = st.radio("Chọn nguồn cấp ảnh:", ("Sử dụng Camera trực tiếp", "Tải ảnh lên từ máy (.jpg, .png)"))
    
    final_image = None
    if source_radio == "Sử dụng Camera trực tiếp":
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
        with st.spinner("Đang nhận diện..."):
            try:
                label, confidence = predict_face(final_image)
                
                st.markdown(
                    f"""
                    <div class="predict-box">
                        <h3 style='margin-top:0; color:#1E3A8A;'>KẾT QUẢ</h3>
                        <p style='font-size: 18px;'>Đối tượng: <strong style='color:#EF4444; font-size: 24px;'>{label}</strong></p>
                        <p style='font-size: 16px;'>Độ tin cậy: <strong>{confidence:.2f}%</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(int(min(max(confidence, 0), 100)))
            except Exception as predict_error:
                st.error(f"Lỗi trong quá trình xử lý ảnh nhận diện: {predict_error}")
    else:
        st.info("Vui lòng cung cấp hình ảnh để hệ thống thực hiện nhận diện.")
