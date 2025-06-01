from typing import List, Tuple

import cv2
import numpy as np


# NOTE: for enhanced filtering, calculate rectangle centroid and use Hungarian matching on contour centroid
def detect_contours(image, roi_coords, method='threshold', visualize=False):
    """
    Detect contours within ROI using either thresholding or Canny edge detection.
    Parameters:
        image : input image
        roi_coords : coordinates: [top-left, top-right, bottom-right, bottom-left]
        method : binarize method: 'threshold'/'canny'
        visualize : visualize image
    Returns:
        List of detected contours, contour visualization, image after binarize
    """
    # Validate
    roi_coords = np.array(roi_coords, dtype=np.int32)
    vis_contour = None
    x, y, w, h = cv2.boundingRect(roi_coords)
    roi = image[y:y + h, x:x + w]
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    if method == 'threshold':
        # Optional: morph opening to break thin connections
        # kernel = np.ones((3, 3), np.uint8)
        # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Adaptive Gaussian thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                       blockSize=51,  # kernel size
                                       C=10)  # constant subtracted from mean

    elif method == 'canny':
        # Canny detection, close to join lines
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'threshold' or 'canny'")

    # Find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if visualize:
        vis_contour = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_contour, contours, -1, (0, 255, 0), 2)

    return contours, vis_contour, binary


def filter_rectangles_geometry(contours):
    """
    Rectangle property check: 4 corners, equal diagonals, perpendicular edges, rectangular fill.
    Parameters:
        contours : List of contours
    Returns:
        List of contours that pass geometric rectangle tests
    """
    filtered_contours = []

    for cnt in contours:
        # Check 1: Must have exactly 4 corners
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        # Get the 4 corners
        corners = approx.reshape(4, 2)
        # Check 2: Equal diagonals
        diag1 = np.linalg.norm(corners[0] - corners[2])
        diag2 = np.linalg.norm(corners[1] - corners[3])
        diag_ratio = min(diag1, diag2) / max(diag1, diag2)
        if diag_ratio < 0.9:  # 10% tolerance
            continue

        # Check 3: angles 90 degrees
        angles_ok = True
        for i in range(4):
            # Get three consecutive corners
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            p3 = corners[(i + 2) % 4]
            # Calculate angle on middle coordinate
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

            if abs(angle - 90) > 10:  # 10 degree tolerance
                angles_ok = False
                break
        if not angles_ok:
            continue

        # Check 4: Contour area vs minimum area rectangle
        contour_area = cv2.contourArea(cnt)
        min_rect = cv2.minAreaRect(cnt)
        rect_area = min_rect[1][0] * min_rect[1][1]

        if contour_area < 0.9 * rect_area:  # area >= 90% bounding rect
            continue

        # Passed all checks
        filtered_contours.append(cnt)

    print(f"Geometric filtering: {len(filtered_contours)}/{len(contours)} contours passed")
    return filtered_contours


# TODO: function not tested
def filter_rectangles_metrics(contours, aspect_ratio, estimated_area,
                              estimated_width, estimated_height):
    """
    Filter contours based on expected metrics from database.
    Parameters:
        contours : List of contours
        aspect_ratio : Expected width/height ratio
        estimated_area : Expected rectangle area (px)
        estimated_width : Expected rectangle width (px)
        estimated_height : Expected rectangle height (px)
    Returns:
        filtered_contours : List of passed contours
    """
    filtered_contours = []

    # Define tolerances
    area_tolerance = 0.25
    dimension_tolerance = 0.20
    aspect_tolerance = 0.15

    for cnt in contours:
        # Get bounding rect
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # Check 1: Area within expected range
        area_ratio = area / estimated_area
        if not (1 - area_tolerance < area_ratio < 1 + area_tolerance):
            continue

        # Check 2: Width within expected range
        width_ratio = w / estimated_width
        if not (1 - dimension_tolerance < width_ratio < 1 + dimension_tolerance):
            continue

        # Check 3: Height within expected range
        height_ratio = h / estimated_height
        if not (1 - dimension_tolerance < height_ratio < 1 + dimension_tolerance):
            continue

        # Check 4: Aspect ratio matches
        actual_aspect = w / h
        aspect_diff = abs(actual_aspect - aspect_ratio) / aspect_ratio
        if aspect_diff > aspect_tolerance:
            continue

        # Passed all metric checks
        filtered_contours.append(cnt)

    print(f"Metrics-based filtering: {len(filtered_contours)}/{len(contours)} contours passed")
    return filtered_contours


def get_rectangle_corners(contours, roi_points, brush_thickness=2):
    """
    Extract rectangle coordinates from contours and map back to original image
    Parameters:
        contours : List of rectangle contours
        roi_points :ROI coordinates used during detection
        brush_thickness: Estimated brush thickness (px)
    Returns:
    rectangles : lList of rectangles: [top-left, top-right, bottom-right, bottom-left]
    """
    # Get ROI offset
    roi_points = np.array(roi_points, dtype=np.int32)
    roi_x, roi_y, _, _ = cv2.boundingRect(roi_points)

    rectangles = []
    approx_count = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            # if contour approximates to rectangle
            approx_count += 1
            corners = approx.reshape(4, 2)

            # Sort x and y coordinates separately
            x_sorted = np.sort(corners[:, 0])
            y_sorted = np.sort(corners[:, 1])

            # Calculate edges from mean of coordinates, then crop inward
            offset = int(brush_thickness * 1.5)
            left = np.mean(x_sorted[:2]) + offset
            right = np.mean(x_sorted[2:]) - offset
            top = np.mean(y_sorted[:2]) + offset
            bottom = np.mean(y_sorted[2:]) - offset

        else:
            # Fallback: use bounding rectangle of approx points
            x, y, w, h = cv2.boundingRect(approx)

            # Calculate edges from bounding box specs, then crop inward
            offset = int(brush_thickness * 2.5)
            left = x + offset
            right = x + w - offset
            top = y + offset
            bottom = y + h - offset

        ordered_corners = np.array([
            [left, top],  # top-left
            [right, top],  # top-right
            [right, bottom],  # bottom-right
            [left, bottom]  # bottom-left
        ])
        # Convert to tuples and add ROI offset
        final_corners = [(int(corner[0] + roi_x), int(corner[1] + roi_y)) for corner in ordered_corners]
        rectangles.append(final_corners)

    print(f"{approx_count}/{len(rectangles)} contours directly approximated to 4-points polygon")
    return rectangles


def get_rect_center(coords):
    points = np.array(coords).reshape(4, 2)
    return np.mean(points, axis=0)


def clustering_row_sort(rectangles, row_tolerance=20) -> List[List[List[Tuple[int, int]]]]:
    """Use DBSCAN to sort rectangles by coordinates, then cluster them into rows"""
    # Cluster by y-coordinate with DBSCAN
    centers = np.array([get_rect_center(rect) for rect in rectangles])
    y_coords = centers[:, 1].reshape(-1, 1)

    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=row_tolerance, min_samples=1).fit(y_coords)

    # Iterate through all cluster label (row)
    rows = {}
    for i, label in enumerate(clustering.labels_):
        if label not in rows:
            rows[label] = []
        rows[label].append((rectangles[i], centers[i]))

    # Sort rows by average y-coordinate, then sort within rows by x-coordinate
    sorted_rows = sorted(rows.items(), key=lambda x: np.mean([c[1][1] for c in x[1]]))

    result = []
    for _, row_data in sorted_rows:
        row_sorted = sorted(row_data, key=lambda x: x[1][0])  # Sort by x-coordinate
        row_rectangles = [rect for rect, _ in row_sorted]
        result.append(row_rectangles)

    return result


def greedy_row_sort(rectangles, y_tolerance=0) -> List[List[List[Tuple[int, int]]]]:
    """Use geometric calculation to sort rectangles by coordinates, then cluster them into rows"""
    remaining = rectangles.copy()
    result_rows = []

    while remaining:
        # Find centroid of all rectangles
        centroids = [get_rect_center(rect) for rect in remaining]

        # Top-left rectangle has centroid closest to (0,0)
        distances = [np.linalg.norm(centroid) for centroid in centroids]
        first_idx = np.argmin(distances)
        rect_first = remaining[first_idx]
        first_centroid = centroids[first_idx]

        # If centroid above mean of top-left's centroid and bottom edge, add to row
        first_points = np.array(rect_first).reshape(4, 2)
        bottom_left_y = first_points[3, 1]
        reference_y = (bottom_left_y + first_centroid[1]) / 2

        rects_on_row = [rect_first]
        indices_to_remove = [first_idx]

        for i, (rect, centroid) in enumerate(zip(remaining, centroids)):
            if i == first_idx:
                continue
            if centroid[1] <= reference_y + y_tolerance:
                rects_on_row.append(rect)
                indices_to_remove.append(i)

        # Sort current row by centroid x-coordinate
        rects_with_centroids = [(rect, get_rect_center(rect)) for rect in rects_on_row]
        rects_with_centroids.sort(key=lambda x: x[1][0])
        row_rectangles = [rect for rect, _ in rects_with_centroids]

        # Add this row to results
        result_rows.append(row_rectangles)
        # Remove processed rectangles
        remaining = [rect for i, rect in enumerate(remaining) if i not in indices_to_remove]

    return result_rows


def validate_grid_sorting(sorted_rows: List[List[Tuple]], expected_layout: List[int]):
    """
    Validate that sorting produces expected grid layout.
    Args:
        sorted_rows: result from sorting methods
        expected_layout: expected row lengths, e.g. [3,3,2]
    Returns:
        bool: whether layout matches expectation
    """
    # Calculate actual layout
    actual_layout = [len(row) for row in sorted_rows]

    # Calculate mean y variance across all rows
    all_y_variances = []
    for row_idx, row_rects in enumerate(sorted_rows):
        if len(row_rects) > 1:  # Need at least 2 points for variance
            row_centroids = [get_rect_center(rect) for rect in row_rects]
            y_coords = [c[1] for c in row_centroids]
            y_variance = np.var(y_coords)
            all_y_variances.append(y_variance)

    mean_y_variance = np.mean(all_y_variances) if all_y_variances else 0

    # Print results
    print(f"Verifying layout {actual_layout} with expected layout {expected_layout}")
    print(f"Vertical variance: {mean_y_variance:.3f} px")

    return actual_layout == expected_layout
