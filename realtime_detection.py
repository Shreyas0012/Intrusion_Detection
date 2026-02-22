import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time

# ===== CONFIG =====
MODEL_PATH = "models/activity_model.h5"
POSE_MODEL_PATH = "pose_landmarker.task"
SEQUENCE_LENGTH = 20
THRESHOLD = 0.6

# ===== LOAD MODEL =====
model = tf.keras.models.load_model(MODEL_PATH)

# ===== LOAD POSE MODEL =====
base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)

# ===== CAMERA =====
cap = cv2.VideoCapture(0)
timestamp = 0

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
video_buffer = deque(maxlen=180)  # ~6 seconds buffer
prediction_history = deque(maxlen=5)

last_state = "Normal"

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    video_buffer.append(frame.copy())

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect_for_video(mp_image, timestamp)
    timestamp += 1

    label = "Normal"
    color = (0, 255, 0)

    if result.pose_landmarks:
        landmarks = []
        for lm in result.pose_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z])

        sequence_buffer.append(landmarks)

        if len(sequence_buffer) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(
                np.array(sequence_buffer), axis=0
            )

            prediction = model.predict(input_data, verbose=0)[0][0]
            prediction_history.append(prediction)

            # Use average prediction for stability
            if len(prediction_history) == 5:
                avg_pred = sum(prediction_history) / 5

                if avg_pred > THRESHOLD:
                    current_state = "Abnormal"
                    label = "⚠ UNUSUAL ACTIVITY"
                    color = (0, 0, 255)
                else:
                    current_state = "Normal"

                # Trigger only on state change (Normal → Abnormal)
                if current_state == "Abnormal" and last_state == "Normal":
                    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"anomaly_{timestamp_str}.avi"

                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(
                        filename,
                        fourcc,
                        20.0,
                        (frame.shape[1], frame.shape[0])
                    )

                    for buffered_frame in video_buffer:
                        out.write(buffered_frame)

                    out.release()

                    print("Anomaly saved:", filename)

                last_state = current_state

    cv2.putText(
        frame,
        label,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

    cv2.imshow("Intrusion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()