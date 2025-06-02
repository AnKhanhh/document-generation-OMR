import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# TODO: implement fallback

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
                   confidence_threshold: float = 0.65
                   ) -> Tuple[Dict[int, List[str]], Dict[int, List[Tuple]]]:
    """
    Extract answers from entire answer sheet.
    Returns:
        (answers_dict, flagged_rects_dict)
        - answers_dict: {question_no: ['A', 'B'], ...} for multi-select
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

    question_no = 1
    partial_count = 0
    for rect_idx, (row_idx, col_idx, corners) in enumerate(rect_list):
        if question_no > num_questions:
            break

        # Extract bounding box from corners
        # corners is [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] in clockwise order
        xs = [corner[0] for corner in corners]
        ys = [corner[1] for corner in corners]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        rect_img = img[y1:y2, x1:x2]

        # Analyze all rows (2D structure preserved)
        min_area_threshold = (bubble_radius ** 2 * np.pi) / 3
        results = analyze_rectangle(rect_img, questions_per_group, choices_per_question, min_area_threshold)

        # Determine valid rows for this rectangle
        if rect_idx == last_rect_idx and trailing_questions > 0:
            valid_rows = trailing_questions
        else:
            valid_rows = min(questions_per_group, num_questions - question_no + 1)

        # Process each question row
        for q_row in range(valid_rows):
            row_results = results[q_row]

            # Check confidence for entire row
            if any(cell['confidence'] < confidence_threshold for cell in row_results):
                if rect_idx not in flagged_rects:
                    flagged_rects[rect_idx] = []
                flagged_rects[rect_idx].append((x1, y1, x2, y2))
                question_no += 1
                continue

            # Extract marked choices for this question
            marked_choices = []
            for choice_idx, cell in enumerate(row_results):
                if cell['fill_ratio'] >= filled_threshold:
                    marked_choices.append(chr(ord('A') + choice_idx))
                elif cell['fill_ratio'] >= partial_threshold:
                    # print(f"Warning: Q {question_no}:{chr(ord('A') + choice_idx)} "
                    #       f"partially filled ({cell['fill_ratio']:.2f})")
                    partial_count += 1

            answers[question_no] = marked_choices
            question_no += 1

    print(f"Successfully extracted {len(answers)}/{num_questions} questions,"
          f" {partial_count} bubbles are partially filled,"
          f" {len(flagged_rects)} question groups are flagged")
    return answers, flagged_rects
