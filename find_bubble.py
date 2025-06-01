import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
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
                           min_size: float = 0.2,
                           circularity_threshold: float = 0.7,
                           binary_threshold: int = -1) -> List[BubbleContour]:
    """
    Find and analyze bubble contours
    Args:
        img: Input grayscale
        min_size: Min fraction of image area for valid bubble
        circularity_threshold: Minimum circularity (4π×area/perimeter²)
        binary_threshold: Fixed threshold value (or -1 for Otsu)

    Returns:
        List of BubbleContour objects with fill_ratio calculated
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Simple binary threshold to preserve actual fill levels
    if binary_threshold == -1:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY_INV)

    # Find all contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return []

    img_area = img.shape[0] * img.shape[1]
    min_area = img_area * min_size

    bubbles = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Size filter - only minimum, no maximum
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
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Calculate fill ratio
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        # Count dark pixels within bubble
        bubble_region = cv2.bitwise_and(binary, mask)
        fill_ratio = np.sum(bubble_region > 0) / np.sum(mask > 0)

        bubbles.append(BubbleContour(
            contour=cnt,
            area=area,
            centroid=(cx, cy),
            circularity=circularity,
            fill_ratio=fill_ratio
        ))

    # Return largest bubble (most likely the actual bubble, not noise)
    return sorted(bubbles, key=lambda b: b.area, reverse=True)


# Helper function for analyzing single rectangle
def analyze_rectangle(rect_img: np.ndarray, rows: int, cols: int) -> List[List[Dict]]:
    """
    Analysis of a rectangle, returning bubble data in row structure.

    Returns:
        List of rows, each containing list of cell analysis dicts
    """
    grid = grid_split(rect_img, rows, cols)
    results = []

    for row_cells in grid:
        row_results = []
        for cell in row_cells:
            bubbles = filter_bubble_contours(cell)

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
                   num_questions: int,
                   questions_per_group: int,
                   choices_per_question: int,
                   filled_threshold: float = 0.55,
                   partial_threshold: float = 0.20,
                   confidence_threshold: float = 0.65) -> Tuple[Dict[int, List[str]], Dict[int, List[Tuple]]]:
    """
    Extract answers from entire answer sheet.

    Args:
        img: Grayscale answer sheet image
        rectangles: List of rectangle rows, each containing list of 4-corner tuples
        num_questions: Total number of questions
        questions_per_group: Questions per rectangle
        choices_per_question: Number of choices (A, B, C, etc.)
        filled_threshold: Min fill ratio for marked answer
        partial_threshold: Fill ratio for partial warning
        confidence_threshold: Min confidence to trust detection

    Returns:
        (answers_dict, flagged_rects_dict)
        - answers_dict: {question_num: ['A', 'B'], ...} for multi-select
        - flagged_rects_dict: {rect_idx: [(x1,y1,x2,y2), ...]}
    """
    answers = {}
    flagged_rects = {}

    # Calculate last rectangle info
    trailing_questions = num_questions % questions_per_group
    last_rect_idx = num_questions // questions_per_group

    # Flatten rectangles for easier indexing
    rect_list = [(r_idx, c_idx, corners)
                 for r_idx, row in enumerate(rectangles)
                 for c_idx, corners in enumerate(row)]

    question_num = 1

    for rect_idx, (row_idx, col_idx, corners) in enumerate(rect_list):
        if question_num > num_questions:
            break

        # Extract bounding box from corners
        # corners is [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] in clockwise order
        xs = [corner[0] for corner in corners]
        ys = [corner[1] for corner in corners]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        rect_img = img[y1:y2, x1:x2]

        # Analyze all rows (2D structure preserved)
        results = analyze_rectangle(rect_img, questions_per_group, choices_per_question)

        # Determine valid rows for this rectangle
        if rect_idx == last_rect_idx and trailing_questions > 0:
            valid_rows = trailing_questions
        else:
            valid_rows = min(questions_per_group, num_questions - question_num + 1)

        # Process each question row
        for q_row in range(valid_rows):
            row_results = results[q_row]  # Direct row access

            # Check confidence for entire row
            if any(cell['confidence'] < confidence_threshold for cell in row_results):
                if rect_idx not in flagged_rects:
                    flagged_rects[rect_idx] = []
                flagged_rects[rect_idx].append((x1, y1, x2, y2))
                question_num += 1
                continue

            # Extract marked choices for this question
            marked_choices = []
            for choice_idx, cell in enumerate(row_results):
                if cell['fill_ratio'] >= filled_threshold:
                    marked_choices.append(chr(ord('A') + choice_idx))
                elif cell['fill_ratio'] >= partial_threshold:
                    print(f"Warning: Q {question_num}:{chr(ord('A') + choice_idx)} "
                          f"partially filled ({cell['fill_ratio']:.2f})")

            answers[question_num] = marked_choices
            question_num += 1

    return answers, flagged_rects
