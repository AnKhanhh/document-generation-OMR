import cv2
import numpy as np


class ArUcoDetector:
    def __init__(self):
        pass
        # self.detectorParams = self.initiate_params()

    @staticmethod
    def high_accuracy_param():
        # Accuracy-optimized parameters
        parameters = cv2.aruco.DetectorParameters()
        # Dense sampling of threshold windows for small markers
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 15  # default 23, for small markers
        parameters.adaptiveThreshWinSizeStep = 2  # default 10, for thorough sampling
        parameters.adaptiveThreshConstant = 6  # Default 7, for subtle contrast
        # Lower perimeter rates for small markers
        parameters.minMarkerPerimeterRate = 0.015  # Default 3%, for smaller markers
        parameters.maxMarkerPerimeterRate = 3.0  # Default 4, conservative upper bound
        parameters.polygonalApproxAccuracyRate = 0.02  # Higher precision
        # High-resolution perspective correction
        parameters.perspectiveRemovePixelPerCell = 12  # Double default for accuracy
        parameters.perspectiveRemoveIgnoredMarginPerCell = 0.15  # Conservative margin
        # Stricter error tolerance for clean generated markers
        parameters.maxErroneousBitsInBorderRate = 0.2  # Lower for cleaner borders
        parameters.errorCorrectionRate = 0.5  # Stricter pattern matching
        # Additional accuracy parameters
        parameters.markerBorderBits = 1  # If using 1-bit border (default)
        parameters.minOtsuStdDev = 5.0  # Default: 5.0
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # Sub-pixel refinement
        parameters.cornerRefinementWinSize = 5  # Window size for refinement
        parameters.cornerRefinementMaxIterations = 30  # More iterations for precision
        parameters.cornerRefinementMinAccuracy = 0.01  # Higher accuracy requirement

        return parameters


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

    # Verify rectangular arrangement by checking edge lengths
    def distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    # Calculate the four edge lengths
    edge_lengths = [
        distance(marker_centers[0], marker_centers[1]),  # Top edge
        distance(marker_centers[1], marker_centers[2]),  # Right edge
        distance(marker_centers[2], marker_centers[3]),  # Bottom edge
        distance(marker_centers[3], marker_centers[0])  # Left edge
    ]

    # Calculate the two diagonal lengths
    diagonal1 = distance(marker_centers[0], marker_centers[2])
    diagonal2 = distance(marker_centers[1], marker_centers[3])

    # Check if opposite edges have similar lengths (within 10% tolerance)
    edges_ok = (abs(edge_lengths[0] - edge_lengths[2]) / max(edge_lengths[0], edge_lengths[2]) < 0.1 and
                abs(edge_lengths[1] - edge_lengths[3]) / max(edge_lengths[1], edge_lengths[3]) < 0.1)

    # Check if diagonals have similar lengths (within 10% tolerance)
    diagonals_ok = abs(diagonal1 - diagonal2) / max(diagonal1, diagonal2) < 0.1

    if edges_ok and diagonals_ok:
        print(f"Marker arrangement verified as rectangular")
        print(f"Edge lengths: {[round(l, 2) for l in edge_lengths]}")
        print(f"Diagonal lengths: {round(diagonal1, 2)}, {round(diagonal2, 2)}")
    else:
        print(f"WARNING: Markers don't form a proper rectangle")
        print(f"Edge lengths: {[round(l, 2) for l in edge_lengths]}")
        print(f"Diagonal lengths: {round(diagonal1, 2)}, {round(diagonal2, 2)}")

    # Calculate the inner corners of each marker (toward the center of document)
    inner_corners = []
    for i, corners_array in enumerate(marker_corners):
        # Get the inner corner depending on marker position
        if i == 0:  # Top-left marker - bottom-right corner is inner
            inner_corners.append(corners_array[2])
        elif i == 1:  # Top-right marker - bottom-left corner is inner
            inner_corners.append(corners_array[3])
        elif i == 2:  # Bottom-right marker - top-left corner is inner
            inner_corners.append(corners_array[0])
        elif i == 3:  # Bottom-left marker - top-right corner is inner
            inner_corners.append(corners_array[1])

    return inner_corners
