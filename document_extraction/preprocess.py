import cv2
import numpy as np


class DocumentPreprocessor:
    def __init__(self, template: np.ndarray):
        self.template = template

    def preprocess_images(self, photo_img, fix_light=False):
        """
        Preprocess images for optimal matching
        """
        assert len(self.template.shape) == 3 and len(photo_img.shape) == 3, "Preprocessing pipeline expects 2 BGR images"

        # Convert to grayscale
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        photo_gray = cv2.cvtColor(photo_img, cv2.COLOR_BGR2GRAY)

        # For photographed image: enhance contrast, denoise, normalize
        if fix_light:
            # larger grid for aggressive lighting balance
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

            # directional lighting correction
            # Calculate and correct illumination field using morphological operations
            kernel_size = max(photo_gray.shape) // 10
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            illumination_field = cv2.morphologyEx(photo_gray, cv2.MORPH_CLOSE, kernel)
            photo_corrected = cv2.divide(photo_gray, illumination_field, scale=255)

            photo_enhanced = clahe.apply(photo_corrected)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            photo_enhanced = clahe.apply(photo_gray)

        # Denoise
        photo_denoised = cv2.bilateralFilter(photo_enhanced, 9, 75, 75)  # (bilateral > Gaussian)
        # normalize
        photo_normalized = cv2.normalize(photo_denoised, None, 0, 255, cv2.NORM_MINMAX)

        return template_gray, photo_normalized
