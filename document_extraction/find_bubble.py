import numpy as np
import cv2
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class BubbleContour:
    """Contour with computed properties"""
    contour: np.ndarray
    area: float
    centroid: Tuple[float, float]
    circularity: float
    fill_ratio: float  # Key metric - how much of bubble is filled


def grid_split(img: np.ndarray, num_row: int, num_col: int,
               overlap: float = 0.1) -> List[List[np.ndarray]]:
    """
    Split rectangle into grid of cells, preserving row structure
    Args:
        img: Input rectangle image
        num_row: Grid info
        num_col: Grid info
        overlap: Fraction of cell to overlap
    Returns:
        List of rows, each containing list of cell images
    """
    h, w = img.shape[:2]
    cell_h, cell_w = h / num_row, w / num_col

    # Overlap to capture bubble
    pad_h = int(cell_h * overlap / 2)
    pad_w = int(cell_w * overlap / 2)

    grid = []
    for r in range(num_row):
        row_cells = []
        for c in range(num_col):
            # Calculate bounds with overlap
            y1 = max(0, int(r * cell_h - pad_h))
            y2 = min(h, int((r + 1) * cell_h + pad_h))
            x1 = max(0, int(c * cell_w - pad_w))
            x2 = min(w, int((c + 1) * cell_w + pad_w))

            row_cells.append(img[y1:y2, x1:x2])
        grid.append(row_cells)

    return grid


def filter_bubble_contours(img: np.ndarray,
                           min_area: int,
                           circularity_threshold: float = 0.7) -> List[BubbleContour]:
    """
    Find and analyze bubble contours
    Args:
        img: Input grayscale
        min_area: min area for valid contour (px^2)
        circularity_threshold: Minimum circularity (4*pi*×area / perimeter ** 2)
    Returns:
        List of BubbleContour objects with fill_ratio calculated
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find all contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []

    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Area check
        if area < min_area:
            continue

        # Circularity check
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < circularity_threshold:
            continue

        # Compute centroid
        moment = cv2.moments(cnt)
        if moment["m00"] == 0:
            continue
        cx = moment["m10"] / moment["m00"]
        cy = moment["m01"] / moment["m00"]

        # Calculate fill ratio
        cnt_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

        shaded_mask = cv2.bitwise_and(binary, cnt_mask)
        fill_ratio = np.sum(shaded_mask > 0) / np.sum(cnt_mask > 0)

        bubbles.append(BubbleContour(
            contour=cnt,
            area=area,
            centroid=(cx, cy),
            circularity=circularity,
            fill_ratio=fill_ratio
        ))

    # Sort bubbles by area
    return sorted(bubbles, key=lambda b: b.area, reverse=True)


# Helper function for analyzing single rectangle
def analyze_rectangle(rect_img: np.ndarray, rows: int, cols: int, min_area: int) -> List[List[Dict]]:
    """
    Analysis of a rectangle, returning bubble data in row structure.
    Returns:
        List of rows, each containing list of cell
    """
    grid = grid_split(rect_img, rows, cols)
    results = []

    for row_cells in grid:
        row_results = []
        for cell in row_cells:
            bubbles = filter_bubble_contours(cell, min_area)

            if not bubbles:
                row_results.append({
                    'fill_ratio': 0.0,
                    'confidence': 0.0
                })
            else:
                bubble = bubbles[0]  # Largest bubble
                row_results.append({
                    'fill_ratio': bubble.fill_ratio,
                    'confidence': bubble.circularity
                })
        results.append(row_results)

    return results


def extract_answer(img: np.ndarray,
                   rectangles: List[List[List[Tuple[int, int]]]],
                   num_questions: int, questions_per_group: int, choices_per_question: int,
                   bubble_radius: int,
                   filled_threshold: float = 0.6,
                   partial_threshold: float = 0.3,
                   confidence_threshold: float = 0.5
                   ) -> Tuple[Dict[int, List[str]], Dict[int, List[Tuple]]]:
    """
    Extract answers from entire answer sheet with fallback bimodal thresholding.

    Returns:
        (answers_dict, flagged_rects_dict)
        - answers_dict: {question_no: ['A', 'B'], ...} for multi-select
        - flagged_rects_dict: {rect_idx: [(x1,y1,x2,y2), ...]}
    """
    # Phase 1: Calculate all cell fill ratios and bimodal thresholds
    all_fill_ratios = []
    cell_cache = {}  # Cache for phase 2

    # Calculate last rectangle info
    trailing_questions = num_questions % questions_per_group
    last_rect_idx = num_questions // questions_per_group

    # Flatten rectangles list for indexing
    rect_list = [(r_idx, c_idx, corners)
                 for r_idx, row in enumerate(rectangles)
                 for c_idx, corners in enumerate(row)]

    # First pass: collect all fill ratios
    for rect_idx, (row_idx, col_idx, corners) in enumerate(rect_list):
        # Get rectangle bounding box
        xs = [corner[0] for corner in corners]
        ys = [corner[1] for corner in corners]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        rect_img = img[y1:y2, x1:x2]
        grid = grid_split(rect_img, questions_per_group, choices_per_question)

        # Determine valid rows
        if rect_idx == last_rect_idx and trailing_questions > 0:
            valid_rows = trailing_questions
        else:
            valid_rows = questions_per_group

        # Calculate fill ratios for all cells
        rect_ratios = []
        for r in range(valid_rows):
            row_ratios = []
            for c in range(choices_per_question):
                cell = grid[r][c]
                gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                fill_ratio = np.sum(binary > 0) / binary.size

                row_ratios.append(fill_ratio)
                all_fill_ratios.append(fill_ratio)
            rect_ratios.append(row_ratios)

        # Cache for phase 2
        cell_cache[rect_idx] = rect_ratios

    # Calculate bimodal thresholds
    ratios_array = np.array(all_fill_ratios)
    ratios_scaled = (ratios_array * 255).astype(np.uint8)

    # Otsu threshold
    otsu_thresh_val = cv2.threshold(ratios_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    otsu_threshold = otsu_thresh_val / 255.0

    # Triangle threshold
    triangle_thresh_val = cv2.threshold(ratios_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)[0]
    triangle_threshold = triangle_thresh_val / 255.0

    print(f"Bimodal thresholds - Otsu: {otsu_threshold:.3f}, Triangle: {triangle_threshold:.3f}")

    # Phase 2: Create all three answer dictionaries
    c_answers = {}  # Contour-based
    o_answers = {}  # Otsu-based
    t_answers = {}  # Triangle-based
    flagged_rects = {}

    question_no = 1
    partial_count = 0
    min_area_threshold = (bubble_radius ** 2 * np.pi) / 3

    for rect_idx, (row_idx, col_idx, corners) in enumerate(rect_list):
        if question_no > num_questions:
            break

        xs = [corner[0] for corner in corners]
        ys = [corner[1] for corner in corners]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        rect_img = img[y1:y2, x1:x2]

        # Contour analysis
        contour_results = analyze_rectangle(rect_img, questions_per_group,
                                            choices_per_question, min_area_threshold)

        # Get cached fill ratios
        cached_ratios = cell_cache[rect_idx]

        # Determine valid rows
        if rect_idx == last_rect_idx and trailing_questions > 0:
            valid_rows = trailing_questions
        else:
            valid_rows = min(questions_per_group, num_questions - question_no + 1)

        # Process each question
        for q_row in range(valid_rows):
            if question_no > num_questions:
                break

            # Contour-based detection
            row_results = contour_results[q_row]
            low_confidence = any(cell['confidence'] < confidence_threshold for cell in row_results)

            if low_confidence:
                if rect_idx not in flagged_rects:
                    flagged_rects[rect_idx] = []
                flagged_rects[rect_idx].append((x1, y1, x2, y2))
                # Skip adding to c_answers
            else:
                marked_choices = []
                for choice_idx, cell in enumerate(row_results):
                    if cell['fill_ratio'] >= filled_threshold:
                        marked_choices.append(chr(ord('A') + choice_idx))
                    elif cell['fill_ratio'] >= partial_threshold:
                        partial_count += 1
                c_answers[question_no] = marked_choices

            # Bimodal-based detection (always process)
            row_ratios = cached_ratios[q_row]

            # Otsu-based
            o_choices = []
            for choice_idx, ratio in enumerate(row_ratios):
                if ratio >= otsu_threshold:
                    o_choices.append(chr(ord('A') + choice_idx))
            o_answers[question_no] = o_choices

            # Triangle-based
            t_choices = []
            for choice_idx, ratio in enumerate(row_ratios):
                if ratio >= triangle_threshold:
                    t_choices.append(chr(ord('A') + choice_idx))
            t_answers[question_no] = t_choices

            question_no += 1

    # Phase 3: Merge if needed
    if flagged_rects:
        # Calculate match scores
        successful_questions = list(c_answers.keys())

        o_matches = sum(1 for q in successful_questions
                        if set(c_answers.get(q, [])) == set(o_answers.get(q, [])))
        t_matches = sum(1 for q in successful_questions
                        if set(c_answers.get(q, [])) == set(t_answers.get(q, [])))

        o_score = o_matches / len(successful_questions) if successful_questions else 0
        t_score = t_matches / len(successful_questions) if successful_questions else 0

        print(f"Fallback matching - Otsu: {o_score:.1%}, Triangle: {t_score:.1%}")

        # Select best method
        selected = o_answers if o_score >= t_score else t_answers
        method = "Otsu" if o_score >= t_score else "Triangle"

        # Fill missing entries
        filled_count = 0
        for q_num, choices in selected.items():
            if q_num not in c_answers:
                c_answers[q_num] = choices
                filled_count += 1

        print(f"Fallback to {method} thresholding to fill {filled_count} entries")

    print(f"Successfully extracted {len(c_answers)}/{num_questions} questions,"
          f" {partial_count} bubbles are partially filled,", end=" ")
    if len(flagged_rects) == 0:
        print("no question groups are flagged")
    else:
        print(f"question groups # {list(flagged_rects.keys())} are flagged")

    return c_answers, flagged_rects
