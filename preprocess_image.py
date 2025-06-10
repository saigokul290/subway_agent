# preprocess_image.py
# Enhanced preprocessing for Subway Surfers:  
# 1) Resize to 128×128  
# 2) Convert to grayscale  
# 3) Apply Canny edge detection to highlight obstacles  
# 4) Normalize to [0,1]  

import numpy as np
import cv2

def preprocess_image(img):
    """
    Original image shape: (H, W, 3) BGR or RGB (from pyautogui)
    1) Resize to 128×128×3
    2) Convert to grayscale (0–255)
    3) Canny edge detection (0–255)
    4) Normalize to 0–1 floats
    5) Expand dims to (1, 128, 128)
    """

    # 1) Resize to 128×128 color
    img_resized = cv2.resize(np.array(img), (128, 128))

    # 2) Convert to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 3) Canny edge detection
    #    Thresholds 100 and 200 can be tuned if needed
    edges = cv2.Canny(img_gray, 100, 200)

    # 4) Normalize to [0,1]
    edges = edges.astype(np.float32) / 255.0

    # 5) Return shape (1, 128, 128)
    return np.expand_dims(edges, axis=0)
