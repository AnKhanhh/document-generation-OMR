import math
from typing import Dict, List, Any

import cv2

from DB_bridging.database_bridge import DatabaseBridge
from document_extraction.preprocessing import DocumentPreprocessor
import document_extraction.detection as dtc
import document_extraction.roi as roi
import document_extraction.find_text as text
import document_extraction.find_group as group
from document_extraction.read_qr import parse_qr


def extract(input_img, template=None, visualize=False):
    if len(input_img.shape) != 2:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    viz: Dict[str, Any] = {}

    # 1. correct homography
    preprocessor = DocumentPreprocessor()
    preprocessor.preprocess_images(template, input_img)
    warped_photo = preprocessor.correct_homography()

    # 2. Read qr and fetch
    sheet_id = parse_qr(warped_photo)

    complete_data = DatabaseBridge.get_complete_data(sheet_id)
    if not complete_data:
        print(f"No data found for sheet id: {sheet_id}")
        return
    else:
        print(f"Data queried from DB with for key={complete_data['instance_id']}")

    from DB_bridging.models import StaticMetrics, DynamicMetrics
    dt_ans: List[Dict[str, Any]] = complete_data['answers']
    dt_dyn: DynamicMetrics = complete_data['dynamic_metrics']
    dt_stt: StaticMetrics = complete_data['static_metrics']

    # 3. Find markers as anchor point, extrapolate distances
    marker_corners, marker_ids, _ = dtc.detect_aruco_markers(warped_photo, cv2.aruco.DICT_6X6_250, visualize=visualize)
    marker_order = (dt_stt.top_left, dt_stt.top_right, dt_stt.bottom_right, dt_stt.bottom_left)
    if visualize:
        viz['aruco'] = _.copy()
    marker_size_px = dtc.mean_edge_length(marker_corners)
    rpl_point_px = marker_size_px / dt_stt.marker_size

    # 4. find ROI coords
    content_corners = dtc.verify_document_markers(marker_corners, marker_ids, marker_order)

    brush_px = rpl_point_px * dt_stt.brush_thickness
    line_length = dt_stt.page_width - dt_stt.margin * 2
    txt_qr_ratio = (dt_stt.txt_label_width + dt_stt.txt_field_width + line_length - dt_stt.qr_size) / 2 / line_length

    roi_coords, *_ = roi.find_roi_from_inner(warped_photo, content_corners, txt_qr_ratio, visualize=visualize)
    roi_coords = roi.crop_roi(roi_coords, int(brush_px * 2))
    if visualize:
        viz['content'] = _[0].copy()
        viz['content_opn'] = _[1].copy()

    # 5. find text box, extract text w CCA
    # First detect the text field rectangles
    txt_field_coords, *_ = text.detect_text_boxes(warped_photo,
                                                  roi_corners=roi_coords[0], brush_thickness=brush_px,
                                                  visualize=visualize)
    if visualize:
        viz['field'] = _[1].copy()
        viz['field_opn'] = _[0].copy()
    # Then remove rectangle edges for CCA
    txt_field_img_list, txt_field_coords = text.remove_box_lines(warped_photo, txt_field_coords,
                                                                 brush_thickness=brush_px,
                                                                 margin=dt_stt.txt_field_y_spacing // 2)
    # Estimate typical area of handwritten text. Detect with CCA
    min_txt_area = int(4 * (brush_px * 2) ** 2)
    _, text_bounding_coords = text.detect_text_bounding_boxes(txt_field_img_list, txt_field_coords,
                                                              brush_px, min_txt_area)
    if visualize:
        viz['txt1'] = _[0].copy()
        viz['txt2'] = _[1].copy()
        viz['txt3'] = _[2].copy()

    # 6. Use contour detection to detect answer rectangles
    contours, *_ = group.detect_contours(warped_photo, roi_coords[2], visualize=visualize)
    if visualize:
        viz['contour_raw'] = _[0]

    filtered_contours = group.filter_rectangles_geometry(contours)

    num_group = math.ceil(dt_dyn.num_questions / dt_dyn.questions_per_group)
    num_contour = len(filtered_contours)
    if num_contour == num_group:
        print(f"Detected all {num_contour} rectangles")
    elif num_contour > num_group:
        rect_width_px = rpl_point_px * dt_dyn.choice_width * dt_dyn.choices_per_question
        rect_height_px = rpl_point_px * dt_dyn.questions_per_group * dt_dyn.question_height
        filtered_contours = group.filter_rectangles_metrics(filtered_contours,
                                                            int(rect_width_px / rect_height_px),
                                                            int(rect_width_px * rect_height_px),
                                                            rect_width_px, rect_height_px)
        num_contour = len(filtered_contours)

    else:
        print(f"Only detect {num_contour} contours / {num_group} expected")

    return viz
