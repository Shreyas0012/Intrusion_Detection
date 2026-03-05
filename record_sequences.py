import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

LABEL = "abnormal"   
SEQUENCE_LENGTH = 30
SAVE_PATH = f"dataset/{LABEL}"

os.makedirs(SAVE_PATH, exist_ok=True)

print("Loading MoveNet model...")
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

cap = cv2.VideoCapture(0)

sequence = []
sequence_count = 0

print("Press 's' to start recording")
print("Press 'q' to quit")

recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    img = cv2.resize(frame, (192, 192))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_img = tf.expand_dims(img, axis=0)
    input_img = tf.cast(input_img, dtype=tf.int32)

    outputs = movenet(input_img)
    keypoints = outputs['output_0'].numpy()


    landmarks = keypoints[0][0][:, :2].flatten()

    if recording:
        sequence.append(landmarks)

        if len(sequence) == SEQUENCE_LENGTH:
            file_path = os.path.join(
                SAVE_PATH, f"{LABEL}_{sequence_count}.npy"
            )
            np.save(file_path, np.array(sequence))
            print(f"Saved sequence {sequence_count}")

            sequence = []
            sequence_count += 1
            recording = False


    status_text = "Recording..." if recording else "Press S to record"
    cv2.putText(display_frame, status_text, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Recording", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print("Started recording sequence")
        recording = True
        sequence = []

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()