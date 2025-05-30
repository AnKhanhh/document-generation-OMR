import cv2
import numpy as np
from reportlab.lib.units import cm


class DocumentPreprocessor:
    def __init__(self):
        self.template = None
        self.photo = None
        self.photo_corrected = None

    def binarize_image(self, img):
        """
        Binarize image to enhance structural elements.
        """
        # Convert to grayscale if not already
        if img is None:
            img = self.photo_corrected
        assert len(img.shape) < 3, "cannot binarize image that is not grayscale"

        gray = img.copy()
        _, structure_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return structure_binary

    def preprocess_images(self, template_img, photo_img, fix_light=False):
        """
        Preprocess images for optimal matching
        """
        # Convert to grayscale
        if len(template_img.shape) > 2:
            template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template_img.copy()

        if len(photo_img.shape) > 2:
            photo_gray = cv2.cvtColor(photo_img, cv2.COLOR_BGR2GRAY)
        else:
            photo_gray = photo_img.copy()

        # For photographed image: enhance contrast, denoise, normalize
        if fix_light:
            # larger grid for aggressive lighting balance
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

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
        print(f"Homography inliner ratio: {np.sum(mask)} / {len(matches)}")

        # TODO: import static metric from DB
        from reportlab.lib.pagesizes import A4
        page_width, page_height = A4
        marker_width_ratio = 1.2 * cm / page_width
        # Warp image, upscale for future detection
        height, width = self.template.shape
        marker_size_target = 80
        marker_size_est = width * marker_width_ratio
        if marker_size_est < marker_size_target:  # upscale if marker size is estimated to be too small
            print(f"Markers estimated to be {marker_size_est}px, upscaling for detection")
            height *= marker_size_target / marker_size_est
            width *= marker_size_target / marker_size_est

        self.photo_corrected = cv2.warpPerspective(self.photo, h, (width, height))

        return self.photo_corrected
