from typing import Any

import cv2
import numpy as np
from pyzbar.pyzbar import decode


def parse_qr(image, roi_points=None):
    """
    Decode QR code from a specified region of interest in a homography-corrected image.
    Args:
        image: Grayscale, homography-corrected image as numpy array
        roi_points: List of 4 points ordered clockwise from top-left
    Returns:
        Decoded QR code string or None if decoding fails
    """

    if roi_points is not None:
        # Extract bounding box coordinates
        points = np.array(roi_points)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        # Extract region using simple slicing (no perspective transform needed)
        roi = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    else:
        roi = image.copy()

    # Detect with pyzbar (better than OpenCV)
    decoded_objects = decode(roi)
    if decoded_objects:
        return decoded_objects[0].data.decode('utf-8')

    # Fallback to OpenCV
    qr_detector = cv2.QRCodeDetector()
    data, bbox, _ = qr_detector.detectAndDecode(roi)

    return data if data else None
