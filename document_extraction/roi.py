from typing import List

import cv2
import numpy as np


def find_roi_from_inner(image, coords: List[np.ndarray[2]], ratio=0.6, visualize=False):
    """
    Extract 3 ROIs from answer sheet
    Args:
        image: Original binary image (already homography corrected)
        coords: List of 4 corner coordinates [top-left, top-right, bottom-right, bottom-left]
        ratio: Ratio for dividing the upper region horizontally (default=0.6)
        visualize: Whether to return visualization (default=False)
    Returns:
        List of coordinates for 3 ROIs
        Optional visualization if visualize=True
        Optional debug image (opened binary) if visualize=True
    """
    # Create mask and crop based on provided coordinates
    coords = np.array(coords, dtype=np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [coords], 255)

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(coords)
    cropped = image[y:y + h, x:x + w].copy()
    mask_cropped = mask[y:y + h, x:x + w]

    # Gaussian binarize, then filter for the trapezoidal content region
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    binary = cv2.bitwise_and(binary, binary, mask=mask_cropped)

    # Opening operation, remove all lines below 75% width
    # TODO: instead of strict open morphology, implement post-processing sorting by line length
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.75), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Hough line detection
    lines = cv2.HoughLinesP(h_lines, 1, np.pi / 180, threshold=w // 3, minLineLength=w // 2, maxLineGap=h // 10)

    # Fallback to using 1d vertical histogram
    if lines is None or len(lines) < 2:
        print("Hough line detection failed, falling back to projection profile")
        # Absolute threshold: 80% width occupancy minimum
        abs_threshold = int(0.8 * w)

        # Find contiguous horizontal slices above threshold
        h_proj = np.sum(h_lines, axis=1)
        regions = []
        in_region = False
        start = 0
        for y in range(h):
            if h_proj[y] >= abs_threshold and not in_region:
                # Start of new region
                in_region = True
                start = y
            elif h_proj[y] < abs_threshold and in_region:
                # End of region
                regions.append((start, y - 1))
                in_region = False
        # Edge case: last region extends to image edge
        if in_region:
            regions.append((start, h - 1))

        # Find center of a thick line
        y_positions = [int((start + end) / 2) for start, end in regions]

        if len(y_positions) < 2:
            raise ValueError(f"Line detection failed")
        y_positions = sorted(y_positions)[:2]

    # Successful Hough detection, post-processing
    else:
        # Extract y-coordinates
        y_avg = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                y_avg.append((y1 + y2) // 2)
        # Cluster y-values to find distinct lines
        y_avg = np.array(y_avg)
        y_avg.sort()

        # Clustering: lines within 10 pixels are considered same line
        clusters = []
        current_cluster = [y_avg[0]]
        for i in range(1, len(y_avg)):
            if y_avg[i] - y_avg[i - 1] <= 10:  # Threshold
                current_cluster.append(y_avg[i])
            else:
                clusters.append(int(np.mean(current_cluster)))
                current_cluster = [y_avg[i]]
        if current_cluster:
            clusters.append(int(np.mean(current_cluster)))

        if len(clusters) < 2:
            raise ValueError(f"After distance filtering (10px), find {len(clusters)} line, need 2")

        y_positions = sorted(clusters)[:2]

    # Account for cropping offset
    y1, y2 = y_positions[0] + y, y_positions[1] + y

    # Calculate vertical split based on ratio parameter
    left_width = int((coords[1][0] - coords[0][0]) * ratio)
    x_split = coords[0][0] + left_width

    # Define ROIs in original image
    roi_coords = [
        # ROI 1: Top-left region - text fields
        [
            [coords[0][0], y1],
            [x_split, y1],
            [x_split, y2],
            [coords[0][0], y2]
        ],

        # ROI 2: Top-right region - qrcode
        [
            [x_split, y1],
            [coords[1][0], y1],
            [coords[1][0], y2],
            [x_split, y2]
        ],

        # ROI 3: Bottom region - answer section
        [
            [coords[0][0], y2],
            [coords[1][0], y2],
            [coords[2][0], coords[2][1]],
            [coords[3][0], coords[3][1]]
        ]
    ]

    # Create visualization if requested
    if visualize:
        vis_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Red boundary
        boundary_color = (255, 0, 0)
        for roi in roi_coords:
            roi_pts = np.array(roi, dtype=np.int32)
            cv2.polylines(vis_image, [roi_pts], True, boundary_color, 2)

        # Visualize opening morphology
        h_lines_vis = cv2.cvtColor(h_lines, cv2.COLOR_GRAY2RGB)

        return roi_coords, vis_image, h_lines_vis

    return roi_coords, None, None


def crop_roi(rois, crop_pixels):
    """
    Crop rectangular ROIs inward by a specified number of pixels.
    Args:
        rois: List of ROIs
        crop_pixels: Number of pixels to crop inward from each edge.
    Returns:
        A list with the same shape as rois, containing the cropped ROIs.
    """
    result = []
    adjustments_factor = [(1, 1), (-1, 1), (-1, -1), (1, -1)]

    for roi in rois:
        new_roi = []
        for i, (x, y) in enumerate(roi):
            dx, dy = adjustments_factor[i]
            new_x = x + dx * crop_pixels
            new_y = y + dy * crop_pixels

            if isinstance(x, np.int32):
                new_x, new_y = np.int32(new_x), np.int32(new_y)
            new_roi.append([new_x, new_y])

        result.append(new_roi)

    return result
