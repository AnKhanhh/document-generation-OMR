import cv2
import numpy as np
from scipy.spatial.distance import cdist


def detect_intersection(image, roi_corners, expected_width, expected_height,
                        expected_x_spacing, expected_y_spacing, brush_thickness,
                        num_group):
    """
    Detect rectangles in an image using morphological operations and Hough lines.
    Args:
        image: Input grayscale image
        roi_corners: ROI [top-left, top-right, bottom-right, bottom-left]
        expected_width: Expected rect width (px)
        expected_height: Expected rect height (px)
        expected_x_spacing: Expected horizontal spacing between rectangles
        expected_y_spacing: Expected vertical spacing between rectangles
        brush_thickness: Line thickness in pixels for clustering
        num_group: number of question groups to be detected
    Returns:
        List of rectangle coordinates, each as [top-left, top-right, bottom-right, bottom-left]
    """

    def cluster_lines(lines, orientation, brush_thickness):
        """
        Cluster collinear line segments
        Args:
            lines: Array of line segments
            orientation: 'horizontal'/'vertical'
            brush_thickness: for distance threshold
        Returns:
            List of clustered lines
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
            else:
                # For vertical lines, group by x-coordinate
                avg_x = (x1 + x2) / 2
                min_y, max_y = min(y1, y2), max(y1, y2)
                line_data.append((avg_x, min_y, max_y, x1, x2))

        # Sort by coordinate (y for horizontal, x for vertical)
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

    # noinspection DuplicatedCode
    # code is duplicated for readability
    def merge_cluster(cluster, orientation):
        """Merge line segments in a cluster into a single line"""
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

    # 1. Get bounding ROI
    roi_points = np.array(roi_corners, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(roi_points.astype(np.int32))
    roi_image = image[y:y + h, x:x + w]

    # 2. Binarize
    _, binary = cv2.threshold(roi_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Opening morphology
    # Create kernel to remove features than 3/4 expected dimensions
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(expected_width * 0.75), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(expected_height * 0.75)))

    # Apply kernel to get lines
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Detect line segments
    h_segments = cv2.HoughLinesP(horizontal_lines, 1, np.pi / 180, threshold=50,
                                 minLineLength=expected_width // 2,
                                 maxLineGap=expected_x_spacing // 3)

    v_segments = cv2.HoughLinesP(vertical_lines, 1, np.pi / 180, threshold=50,
                                 minLineLength=expected_height // 2,
                                 maxLineGap=expected_y_spacing // 3)

    # 4. Cluster line segments
    h_lines = cluster_lines(h_segments, 'horizontal', brush_thickness)
    v_lines = cluster_lines(v_segments, 'vertical', brush_thickness)

    # Result validation
    if not h_lines or not v_lines:
        print("Cant detect any lines for q group failed")
        return []
    if len(h_lines) * 2 < num_group or len(v_lines) * 2 < num_group:
        print(f"Detected {len(h_lines)} horizontal/{len(v_lines)} vertical edges, expected {num_group * 2}")
        return []

    # 5. Find intersections
    from scipy.optimize import linear_sum_assignment

    # Flatten line endpoints to coordinate list
    h_points = []
    for h_line in h_lines:
        h_points.extend([h_line[0], h_line[1]])

    v_points = []
    for v_line in v_lines:
        v_points.extend([v_line[0], v_line[1]])

    # Create distance matrix
    distance_matrix = cdist(np.array(h_points), np.array(v_points), 'euclidean')
    # Find optimal assignment
    h_indices, v_indices = linear_sum_assignment(distance_matrix)

    # Extract matches and compute intersections
    intersections = []
    total_distance = 0
    for h_idx, v_idx in zip(h_indices, v_indices):
        h_point = h_points[h_idx]
        v_point = v_points[v_idx]

        # Calculate coord in original image
        intersection_x = (h_point[0] + v_point[0]) / 2
        intersection_y = (h_point[1] + v_point[1]) / 2
        intersections.append((intersection_x + x, intersection_y + y))

        # Accumulate distance for debugging
        total_distance += distance_matrix[h_idx, v_idx]

    # Print debug info
    mean_distance = total_distance / len(h_indices)
    print(f"Detected {len(intersections)}/{num_group * 4} expected")
    print(f"Mean localization accuracy of intersections: {mean_distance:.2f}px")

    return intersections


def intersections_to_rectangles(intersections):
    """
    Group intersection points into rectangles using closest neighbor approach.
    Args:
        intersections: List of intersection coordinates
    Returns:
        List of coordinates: [top-left, top-right, bottom-right, bottom-left]
    """
    rectangles = []
    used_intersections = set()

    # Sort intersections top-left to bottom-right
    sorted_intersections = sorted(intersections, key=lambda p: (p[1], p[0]))

    for i, current_point in enumerate(sorted_intersections):
        if i in used_intersections:
            continue

        # Calculate euclidean distances
        distances = []
        for j, other_point in enumerate(sorted_intersections):
            if j != i and j not in used_intersections:
                dist = np.sqrt((other_point[0] - current_point[0]) ** 2 +
                               (other_point[1] - current_point[1]) ** 2)
                distances.append((j, other_point, dist))

        # Sort by distance and take 3 closest
        distances.sort(key=lambda x: x[2])
        closest_3 = distances[:3]

        if len(closest_3) == 3:
            # Form rectangle from current point + 3 closest
            rect_points = [current_point] + [point[1] for point in closest_3]
            point_indices = [i] + [point[0] for point in closest_3]

            # Sort points in clockwise order from top-left
            clockwise_points = sort_points_clockwise(rect_points)
            rectangles.append(clockwise_points)

            # Mark points as used
            for idx in point_indices:
                used_intersections.add(idx)

    return rectangles


def sort_points_clockwise(points):
    """Sort 4 points in clockwise order starting from top-left."""
    if len(points) != 4:
        print(f"Warning: trying to sort {len(points)} coordinates, expected 4")
    sorted_points = sorted(points, key=lambda p: (p[1], p[0]))  # Sort by y, then x
    return sorted_points
