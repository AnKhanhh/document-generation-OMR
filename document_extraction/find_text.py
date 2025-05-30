import cv2
import numpy as np


def detect_text_boxes(image, roi_corners, brush_thickness, visualize=False):
    """
    Detect three text boxes within an ROI using morphological operations and line detection.
    Args:
        image: Grayscale image
        roi_corners: List of ROI coordinates (clockwise from top-left)
        brush_thickness: Thickness of the lines in pixels
        visualize: Whether to return visualization images
    Returns:
        List of 3 text boxes coordinates from top to bottom
        If visualize=True, also returns tuple of (morph_image, annotated_image)
    """
    roi_corners = np.array(roi_corners, dtype=np.int32)
    assert len(roi_corners) == 4, "Textbox detection failed: Expected 4 coordinates as input"

    # Initialize visualization images
    morph_image = None
    annotated_image = None

    # Make sure ROI is a perfect rectangle
    x_min, y_min = np.min(roi_corners, axis=0)
    x_max, y_max = np.max(roi_corners, axis=0)
    roi = image[y_min:y_max, x_min:x_max].copy()

    # Binarize
    _, binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological opening, keep horizontal lines above half ROI width
    roi_width = x_max - x_min
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (roi_width // 2, 1))
    morph_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, horizontal_kernel)

    # Create visualization images if requested
    if visualize:
        morph_image = cv2.cvtColor(morph_roi, cv2.COLOR_GRAY2BGR)
        annotated_image = image.copy()
        if len(image.shape) == 2:  # Convert to BGR if grayscale
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)

    # Detect lines
    lines = cv2.HoughLinesP(morph_roi,
                            rho=1, theta=np.pi / 180, threshold=50,  # Algorithm parameter
                            minLineLength=roi_width // 3,
                            maxLineGap=brush_thickness * 3
                            )

    # Check if we found any lines
    if lines is None or len(lines) < 6:
        print(f"Cannot detect text box")
        return [], morph_image, annotated_image

    # Keep only horizontal lines, 20 degree tolerance
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi) % 180
        if angle < 20 or angle > 160:
            horizontal_lines.append((x1, y1, x2, y2, length))

    # Sort by length
    horizontal_lines.sort(key=lambda x: x[4], reverse=True)
    horizontal_lines = horizontal_lines[:min(10, len(horizontal_lines))]

    # Draw detected lines on morph image if visualizing
    if visualize and morph_image is not None:
        for x1, y1, x2, y2, _ in horizontal_lines:
            cv2.line(morph_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Sort by y-midpoint
    y_midpoints = [(y1 + y2) / 2 for x1, y1, x2, y2, _ in horizontal_lines]
    sorted_lines = [(horizontal_lines[i], y_midpoints[i]) for i in range(len(horizontal_lines))]
    sorted_lines.sort(key=lambda x: x[1])

    # Group lines within ( brush_thickness * 2 ) distance
    groups = []
    current_group = [sorted_lines[0][0]] if sorted_lines else []
    current_y = sorted_lines[0][1] if sorted_lines else 0
    for line, y in sorted_lines[1:]:
        if abs(y - current_y) <= brush_thickness * 2:
            # Add to current group
            current_group.append(line)
        else:
            # Start a new group
            if current_group:
                groups.append(current_group)
            current_group = [line]
            current_y = y
    # Add the last group if exist
    if current_group:
        groups.append(current_group)

    # Take 6 longest horizontal line to create 3 boxes
    groups.sort(key=lambda g: sum(line[4] for line in g), reverse=True)
    groups = groups[:min(6, len(groups))]

    # Sort groups by y-position (top to bottom)
    groups.sort(key=lambda g: sum(y1 + y2 for x1, y1, x2, y2, _ in g) / (2 * len(g)))

    # Extract box coordinates from the grouped lines
    boxes = []
    for i in range(0, len(groups), 2):
        if i + 1 < len(groups):  # Make sure lines have pair
            top_lines = groups[i]
            bottom_lines = groups[i + 1]
            # Use the longest line in each group
            top_lines.sort(key=lambda line: line[4], reverse=True)
            bottom_lines.sort(key=lambda line: line[4], reverse=True)

            if top_lines and bottom_lines:
                top_line = top_lines[0]
                bottom_line = bottom_lines[0]

                # Extract coordinates
                x1_top, y1_top, x2_top, y2_top, _ = top_line
                x1_bot, y1_bot, x2_bot, y2_bot, _ = bottom_line

                # Ensure left to right ordering
                if x1_top > x2_top:
                    x1_top, y1_top, x2_top, y2_top = x2_top, y2_top, x1_top, y1_top
                if x1_bot > x2_bot:
                    x1_bot, y1_bot, x2_bot, y2_bot = x2_bot, y2_bot, x1_bot, y1_bot

                # Define box corners clockwise: top-left, top-right, bottom-right, bottom-left
                box = [
                    (x1_top, y1_top),  # top-left
                    (x2_top, y2_top),  # top-right
                    (x2_bot, y2_bot),  # bottom-right
                    (x1_bot, y1_bot)  # bottom-left
                ]

                # Adjust coordinates back to original image space
                adjusted_box = [(x + x_min, y + y_min) for x, y in box]
                boxes.append(adjusted_box)

    # Ensure we have exactly 3 boxes
    if len(boxes) != 3:
        print("Cannot match edges into text boxes")

    # Sort boxes by y-coordinate (top to bottom)
    boxes.sort(key=lambda box: box[0][1])

    # Draw detected boxes on the annotated image if visualizing
    if visualize and annotated_image is not None:
        for box in boxes:
            box_np = np.array(box, dtype=np.int32)
            cv2.polylines(annotated_image, [box_np], True, (0, 255, 0), 2)

    return boxes, morph_image, annotated_image


def remove_box_lines(image, text_boxes, brush_thickness, margin=0):
    """
    Isolate text content by removing box lines.
    Args:
        image: Original grayscale image
        text_boxes: List of text boxes (each box is a list of 4 corner points)
        brush_thickness: Thickness of the lines in pixels
        margin: Number of pixels to expand the ROI by to account for text extending outside boxes

    Returns:
        List of processed ROI images with box lines removed, and their corresponding coordinates
    """
    processed_rois = []
    roi_coords = []

    for box in text_boxes:
        # Get bounding rectangle
        box_np = np.array(box, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(box_np)

        # Apply margin to expand the ROI
        x_xpn = max(0, x - margin)
        y_xpn = max(0, y - margin)
        w_xpn = min(image.shape[1] - x_xpn, w + 2 * margin)
        h_xpn = min(image.shape[0] - y_xpn, h + 2 * margin)
        roi = image[y_xpn:y_xpn + h_xpn, x_xpn:x_xpn + w_xpn].copy()
        # Binarize
        _, binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Adjust box coordinates relative to expanded ROI
        rel_box = np.array([
            [margin, margin],
            [w + margin, margin],
            [w + margin, h + margin],
            [margin, h + margin]
        ], dtype=np.int32)

        # Create mask of box edges (3x brush thickness for redundancy)
        edge_mask = np.zeros_like(binary_roi)
        mask_thickness = int(brush_thickness * 3)
        cv2.polylines(edge_mask, [rel_box], True, 255, mask_thickness)

        # 1. Apply mask to isolate edge area
        edges_only = cv2.bitwise_and(binary_roi, edge_mask)

        # 2. Apply morphological opening to detect box edges in masked area
        # Vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(brush_thickness * 1.5)))
        vertical_lines = cv2.morphologyEx(edges_only, cv2.MORPH_OPEN, vertical_kernel)
        # Horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(brush_thickness * 1.5), 1))
        horizontal_lines = cv2.morphologyEx(edges_only, cv2.MORPH_OPEN, horizontal_kernel)
        # Combine
        all_lines = cv2.bitwise_or(vertical_lines, horizontal_lines)

        # 3. Subtract box edges from the binary image
        result = cv2.subtract(binary_roi, all_lines)

        # 4. Create an influence zone around the lines for closing operation
        # Dilate the lines slightly to create a region where closing will be applied
        influence_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        influence_zone = cv2.dilate(all_lines, influence_kernel, iterations=1)

        # 5. Create a version with closing morphology applied
        closing_size = max(2, brush_thickness // 3)
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (closing_size, closing_size))
        closed_full = cv2.morphologyEx(result, cv2.MORPH_CLOSE, closing_kernel)

        # In the influence zone, overwrite with pixel values of closed image
        influence_zone_inv = cv2.bitwise_not(influence_zone)
        final_result = cv2.bitwise_and(result, influence_zone_inv)
        closed_region = cv2.bitwise_and(closed_full, influence_zone)
        final_result = cv2.bitwise_or(final_result, closed_region)

        # Create coordinates
        corner_coords = [
            (x_xpn, y_xpn),  # top-left
            (x_xpn + w_xpn, y_xpn),  # top-right
            (x_xpn + w_xpn, y_xpn + h_xpn),  # bottom-right
            (x_xpn, y_xpn + h_xpn)  # bottom-left
        ]

        # Append the processed ROI and its coordinates
        processed_rois.append(final_result)
        roi_coords.append(corner_coords)

    return processed_rois, roi_coords


def detect_text_bounding_boxes(text_rois, roi_coordinates, brush_thickness, min_component_area=None):
    """
    Detect text bounding boxes using Connected Component Analysis.
    Args:
        text_rois: List of ROI images with text (after box line removal)
        roi_coordinates: List of ROI coordinates in the original image
        brush_thickness: Thickness of the box lines in pixels
        min_component_area: Minimum area for a component to be considered text
    Returns:
        text_boxes: List of text bounding box images
        text_coords: List of text bounding box coordinates in the original image
    """
    # Minimum text area: area = k * stroke_width^2, with k = 3 - 9, recommended k = 5
    # brush is 1 pt = 0.36 mm
    # handwriting = 0.5 mm - 1 mm
    # printing = 0.2 - 0.7 mm
    if min_component_area is None:
        min_component_area = 5 * (brush_thickness ** 2)
        min_component_area = int(min_component_area)

    text_boxes = []
    text_coords = []

    for roi_idx, (roi, roi_coord) in enumerate(zip(text_rois, roi_coordinates)):
        # Get the ROI top-left corner
        roi_x, roi_y = roi_coord[0]

        # Apply connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            roi, connectivity=8
        )

        # Filter based on area, skip background label 0
        valid_components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]

            # Basic size filtering
            if area < min_component_area:
                continue

            # Aspect ratio filtering (exclude extremely thin horizontal/vertical lines)
            aspect_ratio = width / max(height, 1)
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                continue

            # Density filtering
            # - Printed text: 0.3-0.5
            # - Handwritten text: 0.15-0.35
            # - Line artifacts: below 0.15
            # Sources: ICDAR competitions, Smith's "Document Understanding"
            density = area / (width * height)
            if density < 0.2:
                continue

            # Add valid component
            valid_components.append({
                'label': i,
                'x': stats[i, cv2.CC_STAT_LEFT],
                'y': stats[i, cv2.CC_STAT_TOP],
                'w': width,
                'h': height,
                'area': area,
                'centroid': centroids[i]
            })

        # Group components into lines based on y-coord
        if valid_components:
            valid_components.sort(key=lambda c: c['y'])
            text_lines = []
            current_line = [valid_components[0]]
            mean_height = valid_components[0]['h']

            for comp in valid_components[1:]:
                prev_comp = current_line[-1]
                y_diff = abs(comp['y'] - prev_comp['y'])
                if y_diff < mean_height * 0.7:
                    # If y difference is small, cluster
                    current_line.append(comp)
                    mean_height = sum(c['h'] for c in current_line) / len(current_line)
                else:
                    # Start a new line
                    text_lines.append(current_line)
                    current_line = [comp]
                    mean_height = comp['h']
            # Add the last line
            if current_line:
                text_lines.append(current_line)

            # Select the most promising text line for each ROI
            # (Form fields typically contain a single line of text)
            if text_lines:
                # Score based on area, density, centrality, and component count
                scored_lines = []
                for line in text_lines:
                    total_area = sum(comp['area'] for comp in line)
                    component_count = len(line)

                    # Calculate average vertical position (centrality)
                    avg_y = sum(comp['centroid'][1] for comp in line) / component_count

                    # Centrality score - higher for lines closer to center of ROI
                    # 1.0 for perfectly centered, decreasing toward edges
                    center_score = 1.0 - abs((avg_y / roi.shape[0]) - 0.5) * 2

                    # Combined score factors
                    score = total_area * center_score * component_count
                    scored_lines.append((score, line))

                # Keep only the highest scoring line
                if scored_lines:
                    best_line = max(scored_lines, key=lambda x: x[0])[1]
                    text_lines = [best_line]

            # For each text line, create a bounding box
            for line in text_lines:
                # Sort components horizontally
                line.sort(key=lambda c: c['x'])

                # Find bounding box for the entire line
                min_x = min(comp['x'] for comp in line)
                min_y = min(comp['y'] for comp in line)
                max_x = max(comp['x'] + comp['w'] for comp in line)
                max_y = max(comp['y'] + comp['h'] for comp in line)

                # Calculate mean height for dynamic padding
                mean_height = sum(comp['h'] for comp in line) / len(line)

                # Dynamic padding based on text height and brush thickness
                padding_y = max(mean_height * 0.75, int(brush_thickness * 2.5))
                padding_x = max(mean_height * 0.25, brush_thickness * 2)

                # Apply padding with ROI boundary checks
                min_x = max(0, min_x - int(padding_x))
                min_y = max(0, min_y - int(padding_y))
                max_x = min(roi.shape[1], max_x + int(padding_x))
                max_y = min(roi.shape[0], max_y + int(padding_y))

                # Extract text region from ROI
                text_region = roi[min_y:max_y, min_x:max_x].copy()

                # Calculate coords in original image
                orig_min_x = roi_x + min_x
                orig_min_y = roi_y + min_y
                orig_max_x = roi_x + max_x
                orig_max_y = roi_y + max_y

                # Bounding box coords [top-left, top-right, bottom-right, bottom-left]
                text_box_coord = [
                    (orig_min_x, orig_min_y),
                    (orig_max_x, orig_min_y),
                    (orig_max_x, orig_max_y),
                    (orig_min_x, orig_max_y)
                ]

                # Add to results
                text_boxes.append(text_region)
                text_coords.append(text_box_coord)

                # Enhancements:
                # 1. Merge nearby text regions
                # 2. Split regions with too much space between components
                # 3. Handle multi-line text detection
                # 4. Add rules based on language characteristics
                # 5. Implement word segmentation within text lines

    return text_boxes, text_coords
