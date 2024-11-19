# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 00:21:48 2024

@author: U S E R
"""
import cv2
import numpy as np
from PIL import Image
import os

# Set Current Working Directory
WORK_DIR = r"D:\Dhea\ML\FaceRecogDir"
DATASET_PATH = os.path.join(WORK_DIR, 'dataset')
TRAINER_PATH = os.path.join(WORK_DIR, 'trainer', 'trainer.yml')
CASCADE_PATH = os.path.join(WORK_DIR, 'haarcascade_frontalface_default.xml')

# Initialize the recognizer and detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(CASCADE_PATH)

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
    face_samples = []
    ids = []

    for image_path in image_paths:
        PIL_img = Image.open(image_path).convert('L')  # Convert to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return face_samples, ids

def train_recognizer():
    print("\n[INFO] Training faces. Please wait...")
    faces, ids = get_images_and_labels(DATASET_PATH)
    recognizer.train(faces, np.array(ids))
    recognizer.write(TRAINER_PATH)
    print(f"\n[INFO] {len(np.unique(ids))} faces trained. Exiting program.")

if __name__ == "__main__":
    train_recognizer()


