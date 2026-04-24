import cv2
from ultralytics import YOLO
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import time
print("Loading Models...")
face_model = YOLO('yolov8n-face-lindevs.onnx', task='detect')
interpreter = tf.lite.Interpreter(model_path="affecnet_phase2_finetuned_v2_lite.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
TARGET_SIZE = (96, 96)
MARGIN_FACTOR = 0.15 
PROCESS_EVERY_N_FRAMES = 3
frame_count = 0
cached_results = []       

cap = cv2.VideoCapture(0)
prev_frame_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time
    frame_count += 1
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        results = face_model.predict(frame, conf=0.5, verbose=False)

        faces_batch = []
        box_coords = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                x_margin, y_margin = int(w * MARGIN_FACTOR), int(h * MARGIN_FACTOR)

                nx1, ny1 = max(0, x1 - x_margin), max(0, y1 - y_margin)
                nx2, ny2 = min(frame.shape[1], x2 + x_margin), min(frame.shape[0], y2 + y_margin)

                face_crop = frame[ny1:ny2, nx1:nx2]
                if face_crop.size > 0:
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, TARGET_SIZE)
                    faces_batch.append(face_resized.astype('float32'))
                    box_coords.append((x1, y1, x2, y2))
        if len(faces_batch) > 0:
            cached_results = []

            for i, face in enumerate(faces_batch):
                input_tensor = np.expand_dims(face, axis=0)
                input_tensor = preprocess_input(input_tensor) 
                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])[0]
                emotion_idx = np.argmax(prediction)
                label = EMOTIONS[emotion_idx]
                confidence = prediction[emotion_idx] * 100
                cached_results.append((box_coords[i], label, confidence))
        else:
            cached_results = []
    for (box, label, conf) in cached_results:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {conf:.0f}%"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("CrowdPulse - Real Time Edge", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()