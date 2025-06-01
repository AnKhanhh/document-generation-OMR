import cv2
import numpy as np


# class ArUcoDetector:
#     def __init__(self):
#         pass
#         self.detectorParams = self.initiate_params()
#
#     @staticmethod
#     def high_accuracy_param():
#         # Accuracy-optimized parameters
#         parameters = cv2.aruco.DetectorParameters()
#         # Dense sampling of threshold windows for small markers
#         parameters.adaptiveThreshWinSizeMin = 3
#         parameters.adaptiveThreshWinSizeMax = 15  # default 23, for small markers
#         parameters.adaptiveThreshWinSizeStep = 2  # default 10, for thorough sampling
#         parameters.adaptiveThreshConstant = 6  # Default 7, for subtle contrast
#         # Lower perimeter rates for small markers
#         parameters.minMarkerPerimeterRate = 0.015  # Default 3%, for smaller markers
#         parameters.maxMarkerPerimeterRate = 3.0  # Default 4, conservative upper bound
#         parameters.polygonalApproxAccuracyRate = 0.02  # Higher precision
#         # High-resolution perspective correction
#         parameters.perspectiveRemovePixelPerCell = 12  # Double default for accuracy
#         parameters.perspectiveRemoveIgnoredMarginPerCell = 0.15  # Conservative margin
#         # Stricter error tolerance for clean generated markers
#         parameters.maxErroneousBitsInBorderRate = 0.2  # Lower for cleaner borders
#         parameters.errorCorrectionRate = 0.5  # Stricter pattern matching
#         # Additional accuracy parameters
#         parameters.markerBorderBits = 1  # If using 1-bit border (default)
#         parameters.minOtsuStdDev = 5.0  # Default: 5.0
#         parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # Sub-pixel refinement
#         parameters.cornerRefinementWinSize = 5  # Window size for refinement
#         parameters.cornerRefinementMaxIterations = 30  # More iterations for precision
#         parameters.cornerRefinementMinAccuracy = 0.01  # Higher accuracy requirement
#
#         return parameters


def detect_aruco_markers(grayscale, aruco_dict, id_list=None, visualize=False):
    """
    Detect Aruco markers of specified dictionary and IDs in the image.
    Args:
        grayscale: Input image
        aruco_dict: Aruco dictionary type (e.g., cv2.aruco.DICT_6X6_250)
        id_list: List of marker IDs to filter for, or None to detect all
        visualize: Whether to return an image with drawn bounding boxes
    Returns:
        filtered_corners: List of corner coordinates filtered by requested IDs
        filtered_ids: List of IDs corresponding to filtered_corners
        vis_image: Visualization image with bounding boxes (if visualize=True)
    """
    assert len(grayscale.shape) == 2, "expected grayscale as input"

    # Create visualization image if requested
    vis_image = None
    if visualize:
        # Convert grayscale to BGR for colored visualization
        vis_image = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

    # Define the aruco dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(grayscale)

    # Filter markers by IDs if needed
    filtered_corners = []
    filtered_ids = []
    if ids is not None and id_list is not None:
        ids = ids.flatten()
        for i, id_val in enumerate(ids):
            if id_val in id_list:
                filtered_corners.append(corners[i])
                filtered_ids.append(id_val)
    elif ids is not None:
        # If no specific IDs requested, keep all detected markers
        filtered_corners = corners
        filtered_ids = ids.flatten()
    else:
        print("Found no matching ArUco markers")

    # Print detection information and draw visualization if requested
    print(f"Detected marker with the following IDs:", end=" ")
    for i, (corner, id_val) in enumerate(zip(filtered_corners, filtered_ids)):
        print(int(id_val), end='\t')
        # Draw bounding box if visualization is requested
        if visualize:
            pts = corner[0].astype(np.int32)
            cv2.polylines(vis_image, [pts], True, (0, 0, 255), 5)
    print()

    return filtered_corners, filtered_ids, vis_image


def verify_document_markers(corners, ids, expected_ids):
    """
    Verify document markers and extract document corners.
    Args:
        corners: List of corner coordinates from detectMarkers
        ids: List of detected marker IDs
        expected_ids: Tuple of 4 marker IDs in clockwise order starting from top-left
    Returns:
        document_corners: List of 4 points representing the inner rectangle corners, or None
    """
    # Convert IDs to list
    if ids is not None:
        ids = ids.flatten() if isinstance(ids, np.ndarray) else list(ids)
    else:
        print("No markers detected")
        return None
    # Match markers
    missing_ids = [id_val for id_val in expected_ids if id_val not in ids]
    if missing_ids:
        print(f"Missing markers with ID: {missing_ids}")
        return None
    else:
        print("All expected markers detected")

    # Extract corners for expected markers in the specified order
    marker_centers = []
    marker_corners = []
    for expected_id in expected_ids:
        idx = list(ids).index(expected_id)
        corner = corners[idx][0]  # Extract coordinate list
        center = np.mean(corner, axis=0)
        marker_centers.append(center)
        marker_corners.append(corner)

    def distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    # Calculate the edge lengths
    edge_lengths = [
        distance(marker_centers[0], marker_centers[1]),  # Top edge
        distance(marker_centers[1], marker_centers[2]),  # Right edge
        distance(marker_centers[2], marker_centers[3]),  # Bottom edge
        distance(marker_centers[3], marker_centers[0])  # Left edge
    ]
    # Calculate diagonal lengths
    diagonal1 = distance(marker_centers[0], marker_centers[2])
    diagonal2 = distance(marker_centers[1], marker_centers[3])

    # Check opposite edge lengths (10% tolerance)
    edges_ok = (abs(edge_lengths[0] - edge_lengths[2]) / max(edge_lengths[0], edge_lengths[2]) < 0.1 and
                abs(edge_lengths[1] - edge_lengths[3]) / max(edge_lengths[1], edge_lengths[3]) < 0.1)

    # Check diagonal lengths (10% tolerance)
    diagonals_ok = abs(diagonal1 - diagonal2) / max(diagonal1, diagonal2) < 0.1

    if edges_ok and diagonals_ok:
        print(f"Marker arrangement verified as rectangular")
    else:
        print(f"WARNING: Markers don't form a proper rectangle")
    print(f"Edge lengths: {[round(float(l), 2) for l in edge_lengths]}")
    print(f"Diagonal lengths: {round(float(diagonal1), 2)}, {round(float(diagonal2), 2)}")

    # Calculate offset to be 1/3rd of the margin
    x_coord = marker_corners[0][0][0]  # top-left marker, top-left corner, x-coord
    offset = max(10, x_coord / 3)  # offset is either a third of the margi+n, or 10 pixels
    offset = x_coord - min(x_coord, max(5, x_coord - offset))  # all corners 5 pixels away from boundary
    # Estimate the corners of inner content with this offset
    inner_corners = []
    for i, corners_array in enumerate(marker_corners):
        corner_point = None
        if i == 0:  # Top-left marker - bottom-left corner
            corner_point = corners_array[3].copy()
            corner_point[0] -= offset
        elif i == 1:  # Top-right marker - bottom-right corner
            corner_point = corners_array[2].copy()
            corner_point[0] += offset
        elif i == 2:  # Bottom-right marker - top-right corner
            corner_point = corners_array[1].copy()
            corner_point[0] += offset
        elif i == 3:  # Bottom-left marker - top-left corner
            corner_point = corners_array[0].copy()
            corner_point[0] -= offset
        inner_corners.append(corner_point)

    return inner_corners


def mean_edge_length(corners):
    """
    Calculate the mean length of all edges of all ArUco markers.

    Parameters:
        corners : List of corners, each element is an array shape (1, 4, 2).
    Returns:
        Mean length of all edges of all markers.
    """
    all_edges = []

    for i, marker in enumerate(corners):
        # Get 4 corners
        pts = marker[0]

        # Calculate edge lengths using vectorized operations
        edges = np.sqrt(np.sum((pts - np.roll(pts, -1, axis=0)) ** 2, axis=1))
        all_edges.extend(edges)

    # Calculate overall statistics
    mean_length = np.mean(all_edges)
    std_length = np.std(all_edges)

    # Print overall diagnostics
    print(f"Analyzing {len(corners)} markers:"
          f" mean edge length approx. {mean_length:.2f} px, std = {std_length:.2f} px")

    return mean_length
