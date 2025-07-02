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
    x_min, y_min = roi_corners.min(axis=0)
    x_max, y_max = roi_corners.max(axis=0)
    roi_width = x_max - x_min
    roi_height = y_max - y_min
    roi = image[y_min:y_max, x_min:x_max]

    # Binarize
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    results = []
    # Try multiple kernel sizes
    for kernel_factor in [0.6, 0.4, 0.3]:
        kernel_width = int(roi_width * kernel_factor)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Enhanced line detection with lower threshold
        lines = cv2.HoughLinesP(morph, 1, np.pi / 180, 30,  # Lower threshold
                                minLineLength=roi_width // 4,  # Shorter min length
                                maxLineGap=brush_thickness * 4)  # Larger gap

        if lines is not None:
            # Filter for horizontal, 15 degree tolerance
            h_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
                if angle < 20 or angle > 160:
                    y_mid = (y1 + y2) / 2
                    length = np.hypot(x2 - x1, y2 - y1)
                    h_lines.append((x1, y1, x2, y2, y_mid, length))

            results.append((len(h_lines), h_lines, morph, kernel_factor))

    # Select best result (most lines detected)
    best_count, h_lines, morph, kernel_factor = max(results, key=lambda x: x[0])
    print(f"Using factor={kernel_factor} for opening kernel, detecting {best_count} horizontal segments")

    # Clustering
    if len(h_lines) >= 5:
        y_positions = np.array([line[4] for line in h_lines]).reshape(-1, 1)

        # Try to find 6 clusters
        from sklearn.cluster import KMeans
        n_clusters = min(6, len(h_lines))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(y_positions)

        # Get best line from each cluster
        clustered_lines = []
        for i in range(n_clusters):
            cluster_lines = [h_lines[j] for j in range(len(h_lines)) if labels[j] == i]
            if cluster_lines:
                best = max(cluster_lines, key=lambda x: x[5])
                clustered_lines.append(best)

        clustered_lines.sort(key=lambda x: x[4])

        print(f"Clustered into {len(clustered_lines)} line groups")
        for i, line in enumerate(clustered_lines):
            print(f"  C #{i + 1}: y={line[4]:.2f}, length={line[5]:.2f} px")

        # Interpolation fallback
        if len(clustered_lines) == 5:
            print("Expected 6, interpolating the missing line...", end=" ")
            gaps = []
            for i in range(len(clustered_lines) - 1):
                gap = clustered_lines[i + 1][4] - clustered_lines[i][4]
                gaps.append((gap, i))

            # Find the largest gap (likely missing line)
            max_gap, idx = max(gaps)
            avg_gap = sum(g[0] for g in gaps) / len(gaps)

            if max_gap > avg_gap * 1.5:
                print(f"Detected gap between line #{idx} and #{idx + 1}: {max_gap:.2f} px,"
                      f" compared to average: {avg_gap:.2f} px")

                # Try focused detection in the gap region
                y_start = int(clustered_lines[idx][4] + avg_gap * 0.3)
                y_end = int(clustered_lines[idx + 1][4] - avg_gap * 0.3)

                if y_end > y_start:
                    gap_roi = morph[y_start:y_end, :]
                    gap_lines = cv2.HoughLinesP(gap_roi, 1, np.pi / 180, 20,
                                                minLineLength=roi_width // 5,
                                                maxLineGap=brush_thickness * 5)

                    if gap_lines is not None and len(gap_lines) > 0:
                        # Add the detected line
                        x1, y1, x2, y2 = gap_lines[0][0]
                        y_mid = (y1 + y2) / 2 + y_start
                        length = np.hypot(x2 - x1, y2 - y1)
                        clustered_lines.insert(idx + 1, (x1, y1 + y_start, x2, y2 + y_start, y_mid, length))
                        print(f"Extrapolation successfully, y={y_mid:.2f}")

        h_lines = clustered_lines

    # TODO: code fallback not tested
    if len(h_lines) < 6:
        print(f"Warning: detected {len(h_lines)}/6 lines needed to form text fields")
        return [], None, None

    # Take first 6 lines, form boxes
    h_lines = h_lines[:6]
    boxes = []
    for i in range(0, 6, 2):
        top = h_lines[i]
        bottom = h_lines[i + 1]
        box = [
            (min(top[0], top[2]) + x_min, top[4] + y_min),
            (max(top[0], top[2]) + x_min, top[4] + y_min),
            (max(bottom[0], bottom[2]) + x_min, bottom[4] + y_min),
            (min(bottom[0], bottom[2]) + x_min, bottom[4] + y_min)
        ]
        boxes.append(box)

    # Visualization
    morph_vis = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR) if visualize else None
    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if visualize and len(image.shape) == 2 \
        else image.copy() if visualize \
        else None

    if visualize and annotated is not None:
        for x1, y1, x2, y2, _, _ in h_lines:
            cv2.line(morph_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for box in boxes:
            cv2.polylines(annotated, [np.array(box, dtype=np.int32)], True, (0, 255, 0), 2)

    return boxes, morph_vis, annotated


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

        # Create mask of box edges
        edge_mask = np.zeros_like(binary_roi)
        mask_thickness = int(brush_thickness * 5)
        cv2.polylines(edge_mask, [rel_box], True, 255, mask_thickness)

        # 1. Apply mask to isolate edge area
        edges_only = cv2.bitwise_and(binary_roi, edge_mask)

        # 2. Apply morphological opening to detect box edges in masked area
        # Vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(brush_thickness * 3)))
        vertical_lines = cv2.morphologyEx(edges_only, cv2.MORPH_OPEN, vertical_kernel)
        # Horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(brush_thickness * 3), 1))
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

        # Apply CCA
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi, connectivity=8)

        # Filter based on area, skip label 0
        valid_components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]

            if area < min_component_area:
                continue

            aspect_ratio = width / max(height, 1)
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                continue

            # Density filtering
            # - Printed text: 0.3-0.5
            # - Handwritten text: 0.15-0.35
            # - Line artifacts: below 0.15
            # ICDAR competitions, Smith's "Document Understanding"
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
            # (Form fields contain a single line of text)
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

                # Keep the highest scoring line
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
                padding_x = max(mean_height * 0.5, brush_thickness * 2)

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

    print(f"Detected {len(text_coords)}/3 text region")
    for i, box in enumerate(text_coords):
        print(f"region #{i + 1}:", end=" ")
        for j, corner in enumerate(box):
            print(f"({corner[0]}, {corner[1]})", end=" ")
        print()
    return text_boxes, text_coords
