"""
Image Preprocessing Module (Enhanced)

Features:
- Supports both file path and image array input
- Aspect ratio preserving resize (letterbox)
- Optional contrast enhancement (CLAHE)
- Optional normalization
- Modular pipeline design
"""

import os
import cv2
import numpy as np


# ---------------------------
# Core Functions
# ---------------------------

def load_image(image_input):
    """Load image from path OR pass-through if already an array"""
    
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")

        img = cv2.imread(image_input)

        if img is None:
            raise ValueError("Failed to read image")

        return img

    elif isinstance(image_input, np.ndarray):
        return image_input

    else:
        raise TypeError("Input must be file path or numpy array")


def resize_with_aspect_ratio(image, target_size=640):
    """Resize image while preserving aspect ratio (letterbox style)"""

    h, w = image.shape[:2]

    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Create padded image
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def enhance_contrast(image):
    """Enhance image contrast using CLAHE"""

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def normalize_image(image):
    """Normalize image to range [0, 1]"""

    return image.astype(np.float32) / 255.0


# ---------------------------
# Full Pipeline
# ---------------------------

def preprocess_image(
    image_input,
    size=640,
    enhance=True,
    normalize=False
):
    """
    Full preprocessing pipeline

    Steps:
    load → resize (aspect ratio) → enhance → normalize
    """

    img = load_image(image_input)

    img = resize_with_aspect_ratio(img, target_size=size)

    if enhance:
        img = enhance_contrast(img)

    if normalize:
        img = normalize_image(img)

    return img