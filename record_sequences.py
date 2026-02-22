import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# ===== CONFIG =====
LABEL = "normal"   # change to "abnormal" when recording abnormal activity
SEQUENCE_LENGTH = 30
SAVE_PATH = f"dataset/{LABEL}"
MODEL_PATH = "pose_landmarker.task"

os.makedirs(SAVE_PATH, exist_ok=True)

# ===== LOAD MODEL =====
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp = 0
sequence = []
sequence_count = 0

print("Press 's' to start recording a sequence")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect_for_video(mp_image, timestamp)
    timestamp += 1

    if result.pose_landmarks:
        landmarks = []
        for lm in result.pose_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z])

        sequence.append(landmarks)

        if len(sequence) == SEQUENCE_LENGTH:
            file_path = os.path.join(
                SAVE_PATH, f"{LABEL}_{sequence_count}.npy"
            )
            np.save(file_path, np.array(sequence))
            print(f"Saved sequence {sequence_count}")
            sequence = []
            sequence_count += 1

    cv2.imshow("Recording", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()