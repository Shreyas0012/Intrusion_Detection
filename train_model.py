import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ===== LOAD DATA =====
normal_path = "dataset/normal"
abnormal_path = "dataset/abnormal"

X = []
y = []

# Load normal data (label 0)
for file in os.listdir(normal_path):
    data = np.load(os.path.join(normal_path, file))
    X.append(data)
    y.append(0)

# Load abnormal data (label 1)
for file in os.listdir(abnormal_path):
    data = np.load(os.path.join(abnormal_path, file))
    X.append(data)
    y.append(1)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# ===== TRAIN TEST SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== BUILD MODEL =====
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(30, 99)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ===== TRAIN =====
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# ===== EVALUATE =====
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# ===== SAVE MODEL =====
os.makedirs("models", exist_ok=True)
model.save("models/activity_model.keras")
print("Model saved!")