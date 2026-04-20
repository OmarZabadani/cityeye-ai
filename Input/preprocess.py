"""
Image Preprocessing Module

This module provides a simple and robust pipeline for preparing images
before feeding them into computer vision or AI models.

Main functionality:
1. Safely loads an image from disk with validation checks.
2. Resizes the image to a fixed target size (default: 640x640),
   ensuring consistency for model input.
3. Optionally enhances image contrast using CLAHE (Contrast Limited
   Adaptive Histogram Equalization) in LAB color space, which improves
   visibility and feature quality under varying lighting conditions.

Pipeline flow:
    load image → validate → resize → (optional) enhance contrast → return processed image

Functions:
- load_resize_image: Handles image loading, validation, and resizing.
- inhance_contrast: Applies CLAHE-based contrast enhancement.
- preprocess_image: Combines all steps into a single preprocessing pipeline.

This module is designed to be reusable and easily integrated into
larger computer vision systems such as object detection or tracking.
"""
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