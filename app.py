import streamlit as st
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image

# --- Streamlit Page Setup ---
st.set_page_config(page_title="AI Image Classifier", layout="wide")
st.title("🎥 Real-Time Image Classification")

st.markdown(
    "This web app uses your **Keras model** to classify what the webcam sees — "
    "with live updates for each frame. Only the latest prediction is displayed 👇"
)

# --- Load Model and Labels ---
@st.cache_resource
def load_my_model():
    model = load_model("keras_model.h5", compile=False)
    labels = [line.strip() for line in open("labels.txt", "r").readlines()]
    return model, labels

model, class_names = load_my_model()

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")
run = st.sidebar.checkbox("Start Webcam")
camera_id = st.sidebar.selectbox("Select Camera", [0, 1], index=0)
st.sidebar.markdown("Press **Stop Webcam** to end the live feed.")

# --- Placeholders for Dynamic Updates ---
frame_placeholder = st.empty()
prediction_placeholder = st.empty()
confidence_placeholder = st.empty()
progress_placeholder = st.empty()

# --- Webcam Logic ---
camera = cv2.VideoCapture(camera_id)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("❌ Unable to access camera. Try another ID.")
        break

    # Convert and prepare image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Prediction
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])

    # --- Update UI in place ---
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    prediction_placeholder.markdown(f"### 🧠 Class: `{class_name[2:].strip()}`")
    confidence_placeholder.markdown(f"**Confidence:** {confidence_score * 100:.2f}%")
    progress_placeholder.progress(int(confidence_score * 100))

else:
    camera.release()
    frame_placeholder.empty()
    prediction_placeholder.empty()
    confidence_placeholder.empty()
    progress_placeholder.empty()
    st.info("✅ Webcam feed stopped.")
