import cv2
import numpy as np


class DocumentPreprocessor:
    def __init__(self):
        self.template = None
        self.photo = None

    def preprocess_images(self, template_img, photo_img, fix_light=False):
        """
        Preprocess images for optimal matching
        """
        assert len(template_img.shape) > 2 and len(photo_img.shape) > 2, "Preprocessing pipeline expects 2 BGR images"

        # Convert to grayscale
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
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

        # Denoise, normalize
        photo_denoised = cv2.bilateralFilter(photo_enhanced, 9, 75, 75)  # (bilateral > Gaussian)
        photo_normalized = cv2.normalize(photo_denoised, None, 0, 255, cv2.NORM_MINMAX)

        self.template = template_gray
        self.photo = photo_normalized

        return template_gray, photo_normalized

    def correct_homography(self):
        # Detect with AKAZE matcher
        akaze = cv2.AKAZE_create()
        kp_t, des_t = akaze.detectAndCompute(self.template, None)
        kp_p, des_p = akaze.detectAndCompute(self.photo, None)

        # Lowe's radio test
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = bf.knnMatch(des_t, des_p, k=2)
        matches = []
        for m, n in raw_matches:
            if m.distance < 0.7 * n.distance:
                matches.append(m)

        # Extract matches location into coordinate lists
        points_t = np.zeros((len(matches), 2), dtype=np.float32)
        points_p = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points_t[i, :] = kp_t[match.queryIdx].pt
            points_p[i, :] = kp_p[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points_p, points_t, cv2.RANSAC, 5.0)
        print(f"Inliner ratio: {np.sum(mask)} / {len(matches)}")

        # Warp image
        height, width = self.template.shape
        warped_photo = cv2.warpPerspective(self.photo, h, (width, height))

        return warped_photo
