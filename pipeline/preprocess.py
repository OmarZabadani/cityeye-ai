"""
Image Preprocessing Module (Production-Ready)

Purpose:
Prepare images consistently for AI models.

Features:
- Accepts file path OR numpy array
- Aspect-ratio preserving resize (letterbox)
- Optional CLAHE enhancement
- Optional normalization
"""

import os
import cv2
import numpy as np


class ImagePreprocessor:
    def __init__(self, target_size=640, enhance=False, normalize=False):
        self.target_size = target_size
        self.enhance = enhance
        self.normalize = normalize

    # ---------------------------
    # Public API
    # ---------------------------
    def process(self, image_input):
        img = self._load(image_input)
        img = self._resize_with_aspect_ratio(img)

        if self.enhance:
            img = self._enhance_contrast(img)

        if self.normalize:
            img = self._normalize(img)

        return img

    # ---------------------------
    # Internal Methods
    # ---------------------------
    def _load(self, image_input):
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

    def _resize_with_aspect_ratio(self, image):
        h, w = image.shape[:2]

        scale = min(self.target_size / w, self.target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        canvas = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        x_offset = (self.target_size - new_w) // 2
        y_offset = (self.target_size - new_h) // 2

        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas

    def _enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def _normalize(self, image):
        return image.astype(np.float32) / 255.0