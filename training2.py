import cv2
import numpy as np
from PIL import Image
import os

# Initialize the face detector
WORK_DIR = r"D:\Dhea\ML\FaceRecogDir"
CASCADE_PATH = os.path.join(WORK_DIR, 'haarcascade_frontalface_default.xml')
detector = cv2.CascadeClassifier(CASCADE_PATH)

def augment_image(image):
    # Example augmentations (add more as needed)
    augmented_images = []
    augmented_images.append(image)  # Original image

    # Rotate
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    augmented_images.append(rotated)

    # Flip
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)

    return augmented_images

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
    face_samples = []
    ids = []

    for image_path in image_paths:
        PIL_img = Image.open(image_path).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face = img_numpy[y:y + h, x:x + w]
            for augmented_face in augment_image(face):
                face_samples.append(augmented_face)
                ids.append(id)

    return face_samples, ids
