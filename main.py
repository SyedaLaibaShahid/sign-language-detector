import cv2
import numpy as np
import os
import tensorflow as tf

IMG_SIZE = 64

model = tf.keras.models.load_model("sign_model.h5")
labels = sorted(os.listdir("dataset"))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    pred = model.predict(np.expand_dims(img, axis=0))[0]
    class_idx = np.argmax(pred)
    confidence = pred[class_idx]
    label = labels[class_idx]

    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()