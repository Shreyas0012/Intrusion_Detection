import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from collections import deque
import time

MODEL_PATH = "models/activity_model.h5"
SEQUENCE_LENGTH = 30
THRESHOLD = 0.6

model = tf.keras.models.load_model(MODEL_PATH)

movenet_model = hub.load(
    "https://tfhub.dev/google/movenet/singlepose/lightning/4"
)
movenet = movenet_model.signatures['serving_default']

cap = cv2.VideoCapture(0)

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
video_buffer = deque(maxlen=150)

print("Press 'q' to quit")

last_saved_time = 0
COOLDOWN_SECONDS = 3

while True:

    ret, frame = cap.read()
    if not ret:
        break

    video_buffer.append(frame.copy())

    img = cv2.resize(frame,(192,192))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_img = tf.expand_dims(img, axis=0)
    input_img = tf.cast(input_img, dtype=tf.int32)

    outputs = movenet(input_img)
    keypoints = outputs['output_0'].numpy()

    landmarks = keypoints[0][0][:,:2].flatten()

    sequence_buffer.append(landmarks)

    if len(sequence_buffer) == SEQUENCE_LENGTH:

        input_data = np.expand_dims(
            np.array(sequence_buffer), axis=0
        )

        prediction = model.predict(input_data, verbose=0)[0][0]

        print("Prediction:", prediction)

        current_time = time.time()

        if prediction > THRESHOLD and \
           current_time - last_saved_time > COOLDOWN_SECONDS:

            last_saved_time = current_time

            cv2.putText(frame,
                        "⚠ UNUSUAL ACTIVITY",
                        (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,0,255),3)

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"anomaly_{timestamp}.avi"

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(
                filename,
                fourcc,
                20.0,
                (frame.shape[1],frame.shape[0])
            )

            for buffered_frame in video_buffer:
                out.write(buffered_frame)

            out.release()

        else:
            cv2.putText(frame,
                        "Normal",
                        (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,255,0),3)

    cv2.imshow("Intrusion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()