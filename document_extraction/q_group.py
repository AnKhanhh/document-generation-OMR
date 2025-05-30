import cv2
import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict


def detect_rectangles(image, roi_corners, expected_width, expected_height,
                      expected_h_spacing, expected_v_spacing, brush_thickness):
    """
    Detect rectangles in an image using morphological operations and Hough lines.
    Args:
        image: Input grayscale image
        roi_corners: ROI [top-left, top-right, bottom-right, bottom-left]
        expected_width: Expected rect width (px)
        expected_height: Expected rect height (px)
        expected_h_spacing: Expected horizontal spacing between rectangles
        expected_v_spacing: Expected vertical spacing between rectangles
        brush_thickness: Line thickness in pixels for clustering
    Returns:
        List of rectangle coordinates, each as [top-left, top-right, bottom-right, bottom-left]
    """

    def cluster_lines(lines, orientation, brush_thickness):
        """
        Cluster collinear line segments into logical lines.

        Args:
            lines: Array of line segments from HoughLinesP
            orientation: 'horizontal' or 'vertical'
            brush_thickness: Brush thickness for distance threshold

        Returns:
            List of clustered lines, each represented by two endpoints
        """
        if lines is None or len(lines) == 0:
            return []

        cluster_threshold = 2 * brush_thickness
        clustered_lines = []

        # Convert lines to more convenient format
        line_data = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if orientation == 'horizontal':
                # For horizontal lines, group by y-coordinate
                avg_y = (y1 + y2) / 2
                min_x, max_x = min(x1, x2), max(x1, x2)
                line_data.append((avg_y, min_x, max_x, y1, y2))
            else:  # vertical
                # For vertical lines, group by x-coordinate
                avg_x = (x1 + x2) / 2
                min_y, max_y = min(y1, y2), max(y1, y2)
                line_data.append((avg_x, min_y, max_y, x1, x2))

        # Sort by primary coordinate (y for horizontal, x for vertical)
        line_data.sort(key=lambda x: x[0])

        # Group lines within threshold distance
        current_cluster = [line_data[0]]

        for i in range(1, len(line_data)):
            if abs(line_data[i][0] - current_cluster[-1][0]) <= cluster_threshold:
                current_cluster.append(line_data[i])
            else:
                # Process current cluster
                if current_cluster:
                    clustered_lines.append(merge_cluster(current_cluster, orientation))
                current_cluster = [line_data[i]]

        # Process final cluster
        if current_cluster:
            clustered_lines.append(merge_cluster(current_cluster, orientation))

        return clustered_lines

    def merge_cluster(cluster, orientation):
        """Merge line segments in a cluster into a single line."""
        if orientation == 'horizontal':
            avg_y = np.mean([line[0] for line in cluster])
            min_x = min([line[1] for line in cluster])
            max_x = max([line[2] for line in cluster])
            return [(min_x, avg_y), (max_x, avg_y)]
        else:  # vertical
            avg_x = np.mean([line[0] for line in cluster])
            min_y = min([line[1] for line in cluster])
            max_y = max([line[2] for line in cluster])
            return [(avg_x, min_y), (avg_x, max_y)]

    # Extract ROI from image
    roi_points = np.array(roi_corners, dtype=np.float32)

    # Get bounding rectangle of ROI
    x, y, w, h = cv2.boundingRect(roi_points.astype(np.int32))
    roi_image = image[y:y + h, x:x + w]

    # Binarize image
    _, binary = cv2.threshold(roi_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create morphological kernels
    # Remove features smaller than 3/4 of expected dimensions
    h_kernel_width = max(1, int(expected_width * 0.75))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_width, 1))

    v_kernel_height = max(1, int(expected_height * 0.75))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_height))

    # Extract horizontal and vertical lines
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Detect line segments using Hough transform
    min_line_length = min(expected_width, expected_height) // 2
    max_line_gap = max(expected_h_spacing, expected_v_spacing) // 3

    h_segments = cv2.HoughLinesP(horizontal_lines, 1, np.pi / 180,
                                 threshold=50, minLineLength=min_line_length,
                                 maxLineGap=max_line_gap)

    v_segments = cv2.HoughLinesP(vertical_lines, 1, np.pi / 180,
                                 threshold=50, minLineLength=min_line_length,
                                 maxLineGap=max_line_gap)

    # Cluster line segments into logical lines
    h_lines = cluster_lines(h_segments, 'horizontal', brush_thickness)
    v_lines = cluster_lines(v_segments, 'vertical', brush_thickness)

    # Find all intersections
    intersections = []
    for h_line in h_lines:
        for v_line in v_lines:
            # Calculate intersection point
            h_y = h_line[0][1]  # y-coordinate of horizontal line
            v_x = v_line[0][0]  # x-coordinate of vertical line

            # Check if intersection is within line segments
            h_x_min, h_x_max = min(h_line[0][0], h_line[1][0]), max(h_line[0][0], h_line[1][0])
            v_y_min, v_y_max = min(v_line[0][1], v_line[1][1]), max(v_line[0][1], v_line[1][1])

            if h_x_min <= v_x <= h_x_max and v_y_min <= h_y <= v_y_max:
                intersections.append((v_x + x, h_y + y))  # Convert back to original image coordinates

    # Sort intersections by position (top-left to bottom-right)
    intersections.sort(key=lambda p: (p[1], p[0]))

    # Group intersections into rectangles
    rectangles = []
    used_intersections = set()

    for i, intersection in enumerate(intersections):
        if i in used_intersections:
            continue

        # Find potential rectangle corners
        potential_corners = []

        for j, other_intersection in enumerate(intersections):
            if j == i or j in used_intersections:
                continue

            dx = abs(other_intersection[0] - intersection[0])
            dy = abs(other_intersection[1] - intersection[1])

            # Check if distances match expected rectangle dimensions (with tolerance)
            width_tolerance = expected_width * 0.3
            height_tolerance = expected_height * 0.3

            if (abs(dx - expected_width) <= width_tolerance and dy <= height_tolerance) or \
                (abs(dy - expected_height) <= height_tolerance and dx <= width_tolerance):
                potential_corners.append((j, other_intersection))

        # Try to form rectangles from potential corners
        if len(potential_corners) >= 3:
            # Sort potential corners to form a proper rectangle
            corners_with_indices = [intersection] + [corner[1] for corner in potential_corners[:3]]
            corner_indices = [i] + [corner[0] for corner in potential_corners[:3]]

            # Sort corners in clockwise order starting from top-left
            corners_with_indices.sort(key=lambda p: (p[1], p[0]))  # Sort by y, then x

            if len(corners_with_indices) >= 4:
                # Take first 4 corners and arrange clockwise from top-left
                sorted_corners = sorted(corners_with_indices[:4], key=lambda p: (p[1], p[0]))

                # Basic rectangle validation
                if len(sorted_corners) == 4:
                    rectangles.append(sorted_corners)
                    for idx in corner_indices[:4]:
                        used_intersections.add(idx)

    return rectangles
