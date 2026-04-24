import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import time

st.set_page_config(page_title="CrowdPulse Cloud Demo", layout="wide")
st.title("🧠 CrowdPulse: Real-Time Cloud Inference")

st.sidebar.title("Controls")
st.sidebar.write("This is the cloud-hosted WebRTC version.")
st.sidebar.write("👉 Click the 'START' button in the main window to connect your camera.")

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
TARGET_SIZE = (96, 96)
MARGIN_FACTOR = 0.15 
PROCESS_EVERY_N_FRAMES = 3 

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_model = None
        self.interpreter = None
        self.frame_count = 0
        self.cached_results = []
        self.current_probabilities = None

    def recv(self, frame):
        # Lazy load ML models directly on the WebRTC thread to prevent memory-leaks and thread SegFaults
        if self.face_model is None:
            import cv2
            from ultralytics import YOLO
            import tensorflow as tf
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
            
            self.cv2 = cv2
            self.preprocess_input = preprocess_input
            self.face_model = YOLO('yolov8n-face-lindevs.onnx', task='detect')
            self.interpreter = tf.lite.Interpreter(model_path="affecnet_phase2_finetuned_v2_lite.tflite")
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

        img = frame.to_ndarray(format="bgr24")
        
        # Guard clause in case cv2 hasn't loaded yet
        if self.face_model is None: return frame

        img_rgb = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
        self.frame_count += 1

        if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            results = self.face_model.predict(img_rgb, conf=0.5, verbose=False)
            self.cached_results = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    x_margin, y_margin = int(w * MARGIN_FACTOR), int(h * MARGIN_FACTOR)

                    nx1, ny1 = max(0, x1 - x_margin), max(0, y1 - y_margin)
                    nx2, ny2 = min(img_rgb.shape[1], x2 + x_margin), min(img_rgb.shape[0], y2 + y_margin)

                    face_crop = img_rgb[ny1:ny2, nx1:nx2]
                    if face_crop.size > 0:
                        face_resized = self.cv2.resize(face_crop, TARGET_SIZE)
                        input_tensor = np.expand_dims(face_resized.astype('float32'), axis=0)
                        input_tensor = self.preprocess_input(input_tensor)

                        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
                        self.interpreter.invoke()
                        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                        emotion_idx = np.argmax(prediction)
                        self.cached_results.append(((x1, y1, x2, y2), EMOTIONS[emotion_idx], prediction[emotion_idx]*100))
                        self.current_probabilities = dict(zip(EMOTIONS, prediction))

        for (box, label, conf) in self.cached_results:
            x1, y1, x2, y2 = box
            self.cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.cv2.putText(img, f"{label} {conf:.0f}%", (x1, y1 - 10), 
                        self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    ctx = webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {"frameRate": {"ideal": 15}}, 
            "audio": False
        },
        async_processing=True
    )

with col2:
    st.subheader("Emotion Probability")
    chart_placeholder = st.empty()
if ctx.state.playing:
    while ctx.state.playing:
        if ctx.video_processor:
            probs = ctx.video_processor.current_probabilities
            if probs is not None:
                chart_placeholder.bar_chart(probs)
        time.sleep(0.1)