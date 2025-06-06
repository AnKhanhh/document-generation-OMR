import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Optional, Dict


def preprocess_roi(image: np.ndarray, coords: np.ndarray,
                   visualize: bool = False) -> Tuple[np.ndarray, dict, Optional[np.ndarray]]:
    """
    Preprocess image: mask, crop, binarize, and apply morphological opening.

    Returns:
        Tuple of (binary_opened_image, metadata, visualization_image)
    """
    # Create mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [coords], 255)

    # Get bounding box
    x, y, w, h = cv2.boundingRect(coords)

    # Crop image and mask
    cropped = image[y:y + h, x:x + w]
    mask_cropped = mask[y:y + h, x:x + w]

    # Convert to grayscale if needed
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped

    # Otsu thresholding - cleaner than adaptive for uniform documents
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply mask
    binary = cv2.bitwise_and(binary, binary, mask=mask_cropped)

    # Morphological opening to remove short lines (< 30% of width)
    kernel_width = int(w * 0.3)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    binary_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    metadata = {
        'offset': (x, y),
        'size': (w, h),
        'mask': mask_cropped
    }

    vis_image = None
    if visualize:
        # Stack grayscale, binary original, binary opened, and mask horizontally
        vis_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        vis_binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        vis_opened = cv2.cvtColor(binary_opened, cv2.COLOR_GRAY2RGB)
        vis_mask = cv2.cvtColor(mask_cropped, cv2.COLOR_GRAY2RGB)

        vis_image = np.hstack([vis_gray, vis_binary, vis_opened, vis_mask])

    return binary_opened, metadata, vis_image


def detect_lines_hough(binary: np.ndarray, metadata: dict,
                       min_line_length_ratio: float = 0.3,
                       visualize: bool = False) -> Tuple[List[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect lines using HoughLinesP with adaptive parameters.

    Returns:
        Tuple of (horizontal_lines, vis_all_lines, vis_horizontal_lines)
    """
    w, h = metadata['size']

    # Adaptive parameters based on image size
    min_line_length = int(w * min_line_length_ratio)
    max_line_gap = int(w * 0.1)
    threshold = max(50, w // 10)

    # Detect lines
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lines is None:
        lines = []
    else:
        lines = [line[0] for line in lines]

    # Filter for roughly horizontal lines (angle < 20 degrees)
    horizontal_lines = []
    for x1, y1, x2, y2 in lines:
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 20 or angle > 160:
            horizontal_lines.append([x1, y1, x2, y2])

    vis_all_lines = None
    vis_horizontal_lines = None

    if visualize:
        # Visualization of all detected lines
        vis_all_lines = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        for x1, y1, x2, y2 in lines:
            cv2.line(vis_all_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for all lines

        # Visualization of horizontal lines only
        vis_horizontal_lines = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        for x1, y1, x2, y2 in horizontal_lines:
            cv2.line(vis_horizontal_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for horizontal

    return horizontal_lines, vis_all_lines, vis_horizontal_lines


def cluster_lines(lines: List[np.ndarray], eps: float = 15,
                  visualize: bool = False) -> List[Dict]:
    """
    Cluster lines by their y-coordinates and merge.

    Returns:
        List of merged line clusters with properties
    """
    if not lines:
        return []

    # Extract y-centers
    y_centers = []
    for x1, y1, x2, y2 in lines:
        y_centers.append((y1 + y2) / 2)

    y_centers = np.array(y_centers).reshape(-1, 1)

    # Cluster by y-position
    clustering = DBSCAN(eps=eps, min_samples=1).fit(y_centers)

    # Merge lines in each cluster
    clusters = []
    for label in set(clustering.labels_):
        if label == -1:  # Skip noise
            continue

        cluster_indices = np.where(clustering.labels_ == label)[0]
        cluster_lines = [lines[i] for i in cluster_indices]

        # Calculate cluster properties
        all_x = []
        all_y = []
        for x1, y1, x2, y2 in cluster_lines:
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

        clusters.append({
            'y_center': np.mean(all_y),
            'x_min': min(all_x),
            'x_max': max(all_x),
            'length': max(all_x) - min(all_x),
            'line_count': len(cluster_lines),
            'lines': cluster_lines
        })

    # Sort by total length (length * line_count)
    clusters.sort(key=lambda c: c['length'] * c['line_count'], reverse=True)

    if visualize:
        print(f"Found {len(clusters)} line clusters:")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i}: y={cluster['y_center']:.1f}, "
                  f"length={cluster['length']}, lines={cluster['line_count']}")

    return clusters


def filter_best_lines(clusters: List[Dict], n_lines: int = 2,
                      metadata: dict = None) -> List[float]:
    """
    Filter and select the best n horizontal lines.
    Selection criteria: Keep lines >= 90% of longest, then take topmost n lines.

    Returns:
        Y-coordinates of selected lines
    """
    if not clusters:
        raise ValueError("No line clusters found")

    # Find the longest line cluster
    max_length = max(cluster['length'] for cluster in clusters)

    # Keep clusters that are at least 90% of the longest
    length_threshold = max_length * 0.9
    valid_clusters = [c for c in clusters if c['length'] >= length_threshold]

    if len(valid_clusters) < n_lines:
        # Fallback: take the n longest clusters overall
        print(f"Warning: Only {len(valid_clusters)} clusters pass 90% threshold, taking {n_lines} longest clusters")
        clusters_by_length = sorted(clusters, key=lambda c: c['length'], reverse=True)
        valid_clusters = clusters_by_length[:n_lines]

    # Sort by y-position (ascending) and take the topmost n
    valid_clusters.sort(key=lambda c: c['y_center'])
    selected = valid_clusters[:n_lines]

    # Extract y-positions
    y_positions = [c['y_center'] for c in selected]

    return sorted(y_positions)


def create_rois(coords: np.ndarray, y_lines: List[float],
                metadata: dict, ratio: float = 0.6) -> List[np.ndarray]:
    """
    Create ROI coordinates from detected lines.
    Uses ratio to split based on original ROI edges, not line endpoints.

    Returns:
        List of 3 ROI coordinate arrays
    """
    if len(y_lines) < 2:
        raise ValueError(f"Need at least 2 lines, got {len(y_lines)}")

    # Convert to original image coordinates
    offset_x, offset_y = metadata['offset']
    y1 = int(y_lines[0]) + offset_y
    y2 = int(y_lines[1]) + offset_y

    # Calculate horizontal split based on original ROI edges
    # Use the top edge of the ROI for consistent x-coordinates
    left_x = coords[0][0]  # Top-left x
    right_x = coords[1][0]  # Top-right x
    x_split = left_x + int((right_x - left_x) * ratio)

    return [
        # ROI 1: Top-left (text fields)
        np.array([
            [left_x, y1],
            [x_split, y1],
            [x_split, y2],
            [left_x, y2]
        ], dtype=np.int32),

        # ROI 2: Top-right (QR code)
        np.array([
            [x_split, y1],
            [right_x, y1],
            [right_x, y2],
            [x_split, y2]
        ], dtype=np.int32),

        # ROI 3: Bottom (answer section)
        np.array([
            [coords[0][0], y2],
            [coords[1][0], y2],
            coords[2],
            coords[3]
        ], dtype=np.int32)
    ]


def visualize_rois(image: np.ndarray, roi_coords: List[np.ndarray],
                   detected_lines: List[float] = None, metadata: dict = None) -> np.ndarray:
    """
    Create visualization of detected ROIs.

    Returns:
        Visualization image
    """
    vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Draw ROI boundaries
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR
    for roi, color in zip(roi_coords, colors):
        cv2.polylines(vis, [roi], True, color, 2)

    # Draw detected horizontal lines if provided
    if detected_lines and metadata:
        offset_y = metadata['offset'][1]
        for y in detected_lines:
            y_global = int(y) + offset_y
            cv2.line(vis, (0, y_global), (vis.shape[1] - 1, y_global), (0, 255, 255), 1)

    return vis


def find_roi_from_inner(image: np.ndarray, coords: List[np.ndarray],
                        ratio: float = 0.6, visualize: bool = False) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Main function to extract 3 ROIs using HoughLinesP.

    Args:
        image: Input image
        coords: ROI corners [TL, TR, BR, BL]
        ratio: Horizontal split ratio
        visualize: Enable visualization

    Returns:
        Tuple of (roi_coords, visualization_image)
    """
    coords = np.array(coords, dtype=np.int32)

    # Step 1: Preprocess
    binary, metadata, _ = preprocess_roi(image, coords, visualize=False)

    # Step 2: Detect lines
    lines, _, _ = detect_lines_hough(binary, metadata, visualize=False)
    if not lines:
        raise ValueError("No horizontal lines detected on answer sheet")

    # Step 3: Cluster lines
    clusters = cluster_lines(lines, visualize=False)

    # Step 4: Filter best lines
    y_positions = filter_best_lines(clusters, n_lines=2, metadata=metadata)

    # Step 5: Create ROIs
    roi_coords = create_rois(coords, y_positions, metadata, ratio)

    # Step 6: Visualize
    vis_image = None
    if visualize:
        vis_image = visualize_rois(image, roi_coords, y_positions, metadata)

    return roi_coords, vis_image


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
