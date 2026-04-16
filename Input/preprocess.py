import os
import cv2
import numpy as np

def load_resize_image(image_path, target_size=(640, 640)):
    """Load and resize image safely"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Failed to read image")

    return cv2.resize(img, target_size)


def inhance_contrast(image):
    """Enhance image contrast using CLAHE"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def preprocess_image(image_path, size=640, enhance=True):
    """
    Full preprocessing pipeline:
    load → resize → enhance → return clean image
    """

    img = load_resize_image(image_path, target_size=(size, size))

    if enhance:
        img = inhance_contrast(img)

    return img