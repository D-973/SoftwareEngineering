# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 00:22:58 2024

@author: U S E R
"""

import cv2
import numpy as np
import os

# Set Current Working Directory
WORK_DIR = r"D:\Dhea\ML\FaceRecogDir"
TRAINER_PATH = os.path.join(WORK_DIR, 'trainer', 'trainer.yml')
CASCADE_PATH = os.path.join(WORK_DIR, 'haarcascade_frontalface_default.xml')

# Initialize the recognizer and face cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Name related to ids
NAMES = ['None', 'Dhea', 'Atha', 'Ricky']

# Initialize video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

min_w = 0.1 * cam.get(3)
min_h = 0.1 * cam.get(4)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Failed to capture image.")
        break

    img = cv2.flip(img, 1)  # Flip video
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(min_w), int(min_h)))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < 50:
            name = NAMES[id]
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "unknown"
            confidence_text = f"{round(100 - confidence)}%"

        cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('Face Recognition using OpenCV', img)

    # Handle key press
    if cv2.waitKey(10) & 0xFF == 27:  # Press 'ESC' to exit
        break

# Cleanup
print("\n[INFO] Exiting program and cleaning up.")
cam.release()
cv2.destroyAllWindows()
