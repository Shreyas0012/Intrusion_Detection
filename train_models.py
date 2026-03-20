import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

NORMAL_PATH = "dataset/normal"
ABNORMAL_PATH = "dataset/abnormal"

X = []
y = []

for file in os.listdir(NORMAL_PATH):
    data = np.load(os.path.join(NORMAL_PATH, file))
    X.append(data)
    y.append(0)

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

NORMAL_PATH = "dataset/normal"
ABNORMAL_PATH = "dataset/abnormal"

X = []
y = []

EXPECTED_SHAPE = (30,34)

for file in os.listdir(NORMAL_PATH):
    data = np.load(os.path.join(NORMAL_PATH,file))

    if data.shape != EXPECTED_SHAPE:
        print("Skipping:",file,"shape:",data.shape)
        continue

    X.append(data)
    y.append(0)

for file in os.listdir(ABNORMAL_PATH):
    data = np.load(os.path.join(ABNORMAL_PATH,file))

    if data.shape != EXPECTED_SHAPE:
        print("Skipping:",file,"shape:",data.shape)
        continue

    X.append(data)
    y.append(1)

X = np.array(X)
y = np.array(y)

print("Dataset shape:",X.shape)
print("Labels shape:",y.shape)

# ===== TRAIN TEST SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30,34)),

    tf.keras.layers.Conv1D(64,3,activation='relu'),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.LSTM(64),

    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=8,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# ===== SAVE MODEL =====
os.makedirs("models", exist_ok=True)

model.save("models/activity_model.h5")

print("Model saved!")
