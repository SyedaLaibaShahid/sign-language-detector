import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_PATH = "dataset"
IMG_SIZE = 64

data = []
labels = []
label_map = {}

for idx, folder in enumerate(sorted(os.listdir(DATA_PATH))):
    label_map[idx] = folder

print("Loading images...")
for idx, letter in label_map.items():
    folder = os.path.join(DATA_PATH, letter)
    for img_file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, img_file))
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(idx)

data = np.array(data) / 255.0
labels = np.array(labels)

print(f"Total samples: {len(data)}")

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_map), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("Training model...")
model.fit(X_train, y_train, epochs=12, validation_data=(X_test, y_test))

model.save("sign_model.h5")
print("Model saved!")