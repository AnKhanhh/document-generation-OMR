import cv2

from preprocessing import DocumentPreprocessor
import detection as detc


def extract(input_img, template=None):
    if len(input_img.shape) != 2:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # 1. correct homography
    preprocessor = DocumentPreprocessor()
    preprocessor.preprocess_images(template, input_img)
    corrected = preprocessor.correct_homography()

    # 2. Find markers as anchor point
    marker_corners, marker_ids, _ = detc.detect_aruco_markers(corrected, cv2.aruco.DICT_6X6_250, visualize=False)
    content_corners = detc.verify_document_markers(marker_corners,marker_ids, (0, 3, 2, 1))

    marker_size_px = detc.mean_edge_length(marker_corners)
