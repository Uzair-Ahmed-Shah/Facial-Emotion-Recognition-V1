import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import time

st.set_page_config(page_title="CrowdPulse Edge", layout="wide")
st.title("🧠 CrowdPulse: Real-Time Edge Inference")

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
TARGET_SIZE = (96, 96)
MARGIN_FACTOR = 0.15 
PROCESS_EVERY_N_FRAMES = 3 

st.sidebar.title("Controls")
run_camera = st.sidebar.checkbox('Start Webcam Feed')

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    FRAME_WINDOW = st.image([])

with col2:
    st.subheader("Emotion Probability")
    chart_placeholder = st.empty()

if "cap" not in st.session_state:
    st.session_state.cap = None

if run_camera:
    face_model = YOLO('yolov8n-face-lindevs.onnx', task='detect')
    interpreter = tf.lite.Interpreter(model_path="affecnet_phase2_finetuned_v2_lite.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)

    cap = st.session_state.cap
    frame_count = 0
    cached_results = []
    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        frame_count += 1

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            results = face_model.predict(frame_rgb, conf=0.5, verbose=False)
            cached_results = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    x_margin, y_margin = int(w * MARGIN_FACTOR), int(h * MARGIN_FACTOR)

                    nx1, ny1 = max(0, x1 - x_margin), max(0, y1 - y_margin)
                    nx2, ny2 = min(frame_rgb.shape[1], x2 + x_margin), min(frame_rgb.shape[0], y2 + y_margin)

                    face_crop = frame_rgb[ny1:ny2, nx1:nx2]
                    if face_crop.size > 0:
                        face_resized = cv2.resize(face_crop, TARGET_SIZE)
                        input_tensor = np.expand_dims(face_resized.astype('float32'), axis=0)
                        input_tensor = preprocess_input(input_tensor)

                        interpreter.set_tensor(input_details[0]['index'], input_tensor)
                        interpreter.invoke()
                        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

                        emotion_idx = np.argmax(prediction)
                        cached_results.append(((x1, y1, x2, y2), EMOTIONS[emotion_idx], prediction[emotion_idx]*100))

                        chart_placeholder.bar_chart(dict(zip(EMOTIONS, prediction)))

        for (box, label, conf) in cached_results:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_rgb, f"{label} {conf:.0f}%", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame_rgb, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        FRAME_WINDOW.image(frame_rgb)

else:
    if "cap" in st.session_state and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

    st.write("Camera is currently stopped.")