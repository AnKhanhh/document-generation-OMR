import os
import uuid
from datetime import datetime
from typing import Tuple, Optional, Any

from DB_bridging.database_bridge import DatabaseBridge
from document_generation.layout_1_limits import LayoutConstraints, AnswerSheetLayoutValidator

import cv2
import numpy as np
from reportlab.graphics import renderPDF
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


class AnswerSheetGenerator:
    """
    Generator of multiple-choice answer sheets with dynamic layouts.
    """

    # noinspection SpellCheckingInspection
    def __init__(self, output_dir="out/pdf", debug=False, fill_in=False):
        """Initialize generator with default settings."""
        # Page dimensions
        self.page_width, self.page_height = A4

        # Document layout constants
        self.margin = 1 * cm
        self.header_margin = 1.5 * cm
        self.header_line_margin = 2.5 * cm
        self.marker_size = 1.2 * cm
        self.section_label_height = 0.5 * cm
        self.qr_size = 4 * cm

        # Answer section layout constants
        self.bubble_radius = 0.3 * cm
        self.bubble_horizontal_space = 1 * cm
        self.answer_group_top_margin = 0.3 * cm
        self.question_height = 1 * cm
        self.question_number_label_width = 0.8 * cm
        self.lettering_height = 0.4 * cm

        # Configuration
        self.debug = debug
        self.fill_in = fill_in
        self.output_dir = output_dir

        # Metrics are initialized and will be updated, generate document at least once to get full metrics
        from DB_bridging.models import StaticMetrics, DynamicMetrics
        self.static_metrics = StaticMetrics(
            template_id="layout_1",
            page_width=self.page_width,
            page_height=self.page_height,
            margin=self.margin,
            brush_thickness=1,  # implicitly keep thickness=1 throughout this layout
            qr_size=self.qr_size,
            marker_size=self.marker_size,
            section_label_height=self.section_label_height,
            choice_lettering_height=self.lettering_height,
            bubble_radius=self.bubble_radius
        )
        self.dynamic_metrics = DynamicMetrics(static_template=self.static_metrics)

        os.makedirs(output_dir, exist_ok=True)
        pdfmetrics.registerFont(TTFont('kindergarten', 'resources/font/kindergarten.ttf'))
        pdfmetrics.registerFont(TTFont('Cursive-standard', 'resources/font/Cursive-standard.ttf'))
        pdfmetrics.registerFont(TTFont('Cursive-standard-Bold', 'resources/font/Cursive-standard-Bold.ttf'))
        pdfmetrics.registerFont(TTFont('AnthonioScript', 'resources/font/AnthonioScript.ttf'))
        self.bubble_fill_size = self.bubble_radius * np.random.uniform(0.8, 1.05)
        self.fill_font = np.random.choice(["kindergarten", "Cursive-standard", "Cursive-standard-Bold", "Helvetica"])

    def generate_answer_sheet(self,
                              num_questions: int = 30,
                              choices_per_question: int = 4,
                              questions_per_group: int = 5,
                              sheet_id: str = None,
                              filename: str = None,
                              buffer: Any = None) -> None:
        """Generate answer sheet. Returns filepath, sheet_id, custom metadata"""
        # Input validation and initialization
        sheet_id = sheet_id or str(uuid.uuid4())
        timestamp = datetime.now().strftime("%M:%H_%d/%m%Y")
        filename = filename or f"answer_sheet_{timestamp}.pdf"

        # Create canvas and set metadata
        if buffer is not None:
            c = canvas.Canvas(buffer, pagesize=A4)
        else:
            filepath = os.path.join(self.output_dir, filename)
            c = canvas.Canvas(filepath, pagesize=A4)
        c.setTitle(timestamp)

        # Draw all components in sequence
        self._draw_alignment_markers(c)
        header_y = self._draw_header(c)
        form_region_y = self._draw_form_fields(c, header_y)
        self._draw_barcode(c, sheet_id, header_y)
        self._draw_answer_section(c, num_questions, choices_per_question, questions_per_group, form_region_y)
        c.save()

        self.dynamic_metrics.instance_id = sheet_id

        # return filepath, sheet_id

    def _draw_header(self, c: canvas) -> float:
        """Draw the header of the answer sheet."""
        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(self.page_width / 2, self.page_height - self.header_margin, "ANSWER SHEET")

        # Instruction string
        c.setFont("Helvetica", 10)
        if self.debug:
            c.saveState()
            c.setFillColor(colors.red)
            c.drawCentredString(self.page_width / 2, self.page_height - 2 * cm, "Debug mode")
            c.restoreState()
        else:
            c.drawCentredString(self.page_width / 2, self.page_height - 2 * cm,
                                "Fill the bubbles completely using a dark pen or pencil")

        # Draw horizontal line
        c.setStrokeColor(colors.black)
        self._draw_horizontal_line(c, self.page_height - self.header_line_margin)

        return self.page_height - self.header_line_margin

    def _draw_form_fields(self, c: canvas, y_start: float) -> float:
        """Draw the information fields"""
        # Section header
        y_start -= self.section_label_height
        c.setFont("Helvetica", 12)
        c.drawString(self.margin, y_start, "INFORMATION FIELDS")

        y_start -= 1 * cm

        # Define field height and spacing
        field_height = 0.8 * cm
        field_label_width = 3 * cm
        field_width = 10 * cm
        field_y_spacing = 1.25 * cm

        self.static_metrics.txt_field_width = field_width
        self.static_metrics.txt_field_height = field_height
        self.static_metrics.txt_field_y_spacing = field_y_spacing
        self.static_metrics.txt_label_width = field_label_width

        # Common field drawing function
        def draw_field(label, y_pos):
            c.setFont("Helvetica", 10)
            c.drawString(self.margin, y_pos, label)
            c.rect(self.margin + field_label_width, y_pos - 2 * mm, field_width, field_height)

        def fill_field(text, y_pos):
            c.saveState()
            c.setFillColor(colors.blue)
            c.setFont(self.fill_font, np.random.randint(14, 22))
            c.drawString(self.margin + field_label_width + np.random.uniform(0.25, 4.5) * cm,
                         y_pos + np.random.randint(-2, 1) * mm,
                         f"{text}")
            c.restoreState()

        # Draw each field
        draw_field("Student ID:", y_start)
        if self.fill_in:
            fill_field("20XX YYYY", y_start)
        y_start -= field_y_spacing
        draw_field("Class:", y_start)
        if self.fill_in:
            fill_field(f"IT-T{np.random.randint(1, 18)}", y_start)
        y_start -= field_y_spacing
        draw_field("Location:", y_start)
        if self.fill_in:
            fill_field(f"{np.random.choice(list('BCD'))}{np.random.randint(2, 9)}-{np.random.randint(100, 999)}, HUST", y_start)
        y_start -= (field_y_spacing - 5 * mm)

        # Draw horizontal line
        c.setStrokeColor(colors.black)
        self._draw_horizontal_line(c, y_start)

        return y_start

    def _draw_debug_box(self, c: canvas, x: float, y: float, width: float, height: float, label: str = None) -> None:
        """Draw a debug bounding box, label optional."""
        assert self.debug, "debug function invoked unintentionally"

        c.saveState()
        c.setStrokeColor(colors.red)
        c.setLineWidth(0.5)
        c.rect(x, y, width, height, stroke=1, fill=0)
        if label:
            c.setFillColor(colors.red)
            c.setFont("Helvetica", 8)
            c.drawString(x, y + height + 1 * mm, label)
        c.restoreState()

    def _draw_barcode(self, c: canvas, sheet_id: str, y_start: float) -> None:
        """Draw sheet_id QR code, right-aligned."""
        # Define QR code basic parameters
        qr_size = self.qr_size
        qr_x = self.page_width - self.margin - qr_size

        # Create QR code, high correction for better OMR
        qr_code = qr.QrCodeWidget(value=sheet_id, barLevel='H')
        bounds = qr_code.getBounds()
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        # Draw QR code
        d = Drawing(qr_size, qr_size, transform=[qr_size / width, 0, 0, qr_size / height, 0, 0])
        d.add(qr_code)
        renderPDF.draw(d, c, qr_x, y_start - qr_size - 0.5 * cm)

        # When debug, draw label and ID string
        if self.debug:
            c.saveState()
            c.setFillColor(colors.red)
            c.setFont("Helvetica-Bold", 12)
            c.drawCentredString(qr_x + qr_size / 2, y_start, "SHEET ID")
            c.rotate(90)
            c.setFont("Helvetica", 10)
            c.drawString(y_start - 5 * cm, - self.page_width + 5 * mm, f"ID: {sheet_id}")
            c.restoreState()

    def _draw_answer_section(self, c: canvas,
                             num_questions: int, choices_per_question: int, questions_per_group: int,
                             y_start: float) -> float:
        """
        Draw the answer sections, spacing and layout optimized dynamically, params validated externally
        """
        # 1. Define basic dimensions for answer section
        available_height = y_start - (self.margin + self.marker_size + 0.2 * cm)
        available_width = self.page_width - 2 * self.margin
        answer_section_label_height = self.section_label_height
        alphabet_str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # 2. Initialize subsections (answer groups) parameters
        choice_width = self.bubble_horizontal_space
        question_number_width = self.question_number_label_width
        group_width = question_number_width + (choices_per_question * choice_width)
        group_height = self.question_height * questions_per_group
        group_y_gap = min(self.answer_group_top_margin, group_height * 0.4)  # in case grouping is small
        lettering_height = self.lettering_height

        # 3. Calculate layout (no validation - parameters are pre-validated)
        max_groups_allowed_per_row = int(available_width / group_width)
        num_group = (num_questions + questions_per_group - 1) // questions_per_group

        # Calculate optimal row distribution
        max_rows_allowed = int((available_height - answer_section_label_height - lettering_height) /
                               (group_height + group_y_gap))
        group_distribution_on_rows: list = self._equal_bin_packing(num_group, max_rows_allowed, max_groups_allowed_per_row)
        num_group_row = len(group_distribution_on_rows)
        total_height_needed = (num_group_row * (group_height + group_y_gap)) + answer_section_label_height + lettering_height

        # 4. Optimize spacing
        # Assign spare horizontal space
        for i in range(10):
            groups_on_row = group_distribution_on_rows[0]
            minimum_x_space = groups_on_row * group_width
            group_x_gap = (available_width - minimum_x_space) / max(groups_on_row - 1, 1)
            # Limit iteration, and prevent skipping recalculation in last loop
            if i == 9:
                print("Need more time to optimizing horizontal spacing, moving to next step...")
                break
            # If horizontal spacing too large, optimize
            if group_x_gap / group_width > 0.4 or groups_on_row == 1:
                # If there is enough space, each row has 1 more group
                if (group_x_gap * (groups_on_row - 1) > group_width * 1.5
                    and num_group_row > 1
                    and num_group_row > group_distribution_on_rows[num_group_row - 1]):
                    new_distribution = group_distribution_on_rows[:-1].copy()
                    new_distribution.sort()
                    for j in range(group_distribution_on_rows[num_group_row - 1]):
                        if j > len(new_distribution) - 1:
                            print("Error while reassigning groups!")
                            break
                        new_distribution[j] += 1
                    group_distribution_on_rows = new_distribution
                    group_distribution_on_rows.sort(reverse=True)
                    num_group_row = len(group_distribution_on_rows)
                    total_height_needed = (num_group_row * (group_height + group_y_gap)) + answer_section_label_height
                    continue
                # If not enough space for new group, extend choice_width. gap = 0.3 group width is ideal
                elif choice_width < 1.5 * self.bubble_horizontal_space:
                    choice_width += (available_width - (minimum_x_space + group_width * 0.3 * (groups_on_row - 1))) \
                                    / (groups_on_row * choices_per_question)
                    choice_width = min(choice_width, self.bubble_horizontal_space * 1.25)
                    group_width = question_number_width + (choices_per_question * choice_width)
                    continue
                else:
                    break
            else:
                break

        # Assign spare vertical space
        group_y_gap += (available_height - total_height_needed) / num_group_row
        group_y_gap = min(group_y_gap, self.answer_group_top_margin * 2.5)
        total_height_needed = (num_group_row * (group_height + group_y_gap)) + answer_section_label_height

        # 5.Draw answer section
        # Finalize boundary
        section_end_y = max(y_start - total_height_needed, self.margin + self.marker_size)
        section_height = y_start - section_end_y

        if self.debug:
            self._draw_debug_box(c,
                                 self.margin, section_end_y,
                                 available_width, section_height,
                                 "Answer Section Box")

        # Draw section label
        y_start -= answer_section_label_height
        c.setFont("Helvetica", 12)
        c.drawString(self.margin, y_start, "ANSWER SECTION")

        # Draw choices lettering
        y_start -= lettering_height
        c.setFont("Helvetica", 10)
        for column_no in range(num_group_row):
            for choice_no in range(choices_per_question):
                lettering_x = self.margin + column_no * (group_width + group_x_gap) + question_number_width + (choice_no + 0.5) * choice_width
                # y-coord offset for answer group top padding
                c.drawCentredString(lettering_x, y_start - group_y_gap + 2 * mm, alphabet_str[choice_no])

        # Draw answer groups by row
        current_group = 0
        for row in range(num_group_row):
            # Add margin on top of each row
            y_start -= group_y_gap
            # Top y-coord of row after margin
            row_top_y = y_start

            # Draw answer groups on a row
            groups_on_row = group_distribution_on_rows[row]
            for col in range(groups_on_row):
                assert current_group < num_group, "answer group index out of bound"
                try:
                    group_x_gap
                except NameError:
                    group_x_gap = (available_width - groups_on_row * group_width) / max(groups_on_row - 1, 1)
                    print("Error while optimizing space")

                group_x = self.margin + (col * (group_width + group_x_gap))

                # Draw answer group box
                if self.debug:
                    self._draw_debug_box(c, group_x, row_top_y - group_height, group_width, group_height,
                                         f"Group {current_group + 1} Box")
                else:
                    c.rect(group_x + self.question_number_label_width, row_top_y - group_height,
                           group_width - question_number_width, group_height)

                # Calculate question range, draw group
                first_q = (current_group * questions_per_group) + 1
                last_q = min((current_group + 1) * questions_per_group, num_questions)
                self._draw_question_group(c, group_x, row_top_y,
                                          first_q, last_q, self.question_height, question_number_width,
                                          choice_width, choices_per_question)
                current_group += 1

            y_start -= group_height

        if y_start < (self.margin + self.marker_size):
            print("Unexpected behavior: Answer section drawn out of bound.")

        self.dynamic_metrics.question_height = self.question_height
        self.dynamic_metrics.choice_width = choice_width
        self.dynamic_metrics.group_y_spacing = group_y_gap
        self.dynamic_metrics.group_x_spacing = group_x_gap
        self.dynamic_metrics.num_questions = num_questions
        self.dynamic_metrics.questions_per_group = questions_per_group
        self.dynamic_metrics.choices_per_question = choices_per_question
        self.dynamic_metrics.layout = group_distribution_on_rows

        return y_start

    def _draw_question_group(self, c: canvas, x: float, y: float,
                             first_q: int, last_q: int,
                             question_height: float, question_number_width: float,
                             choice_width: float, choices_per_question: int) -> None:
        """
        Draw a group of questions with answer bubbles.
        """
        c.saveState()
        c.setLineWidth(1)
        answer_y = y

        for q_no in range(first_q, last_q + 1):
            answer_y -= question_height / 2
            # 1mm offset to align number with bubble
            c.setFont("Helvetica", 10)
            c.drawString(x + 1 * mm, answer_y - 1 * mm, f"{q_no}.")

            if self.debug:
                self._draw_debug_box(c, x, answer_y - question_height / 2, question_number_width, question_height)

            # Use normal distribution to select bubbles
            def generate_answer() -> list[int]:
                num_selections = round(max(0, min(choices_per_question - 1, np.random.normal(1, 0.65))))
                if num_selections == 0:
                    return []
                else:
                    return np.random.choice(range(0, choices_per_question), size=num_selections, replace=False)

            selections = []
            if self.fill_in:
                selections = generate_answer()

            # Draw bubbles
            for choice_no in range(choices_per_question):
                choice_x = x + question_number_width + ((choice_no + 0.5) * choice_width)
                # Circle in ReportLab is drawn from center
                c.circle(choice_x, answer_y, self.bubble_radius, stroke=1, fill=0)
                # Testing code
                if self.fill_in and choice_no in selections:
                    c.circle(choice_x + np.random.uniform(0, 0.2) * self.bubble_fill_size,
                             answer_y + np.random.uniform(0, 0.2) * self.bubble_fill_size,
                             self.bubble_fill_size,
                             stroke=1, fill=1)
                # Debug code
                if self.debug:
                    self._draw_debug_box(c, choice_x - choice_width / 2, answer_y - question_height / 2,
                                         choice_width, question_height)

            answer_y -= question_height / 2
        c.restoreState()

    def _draw_horizontal_line(self, c: canvas, y_position: float) -> None:
        """Draw a straight line, account for margin."""
        c.line(self.margin, y_position, self.page_width - self.margin, y_position)

    # noinspection SpellCheckingInspection
    def _draw_alignment_markers(self, c: canvas) -> None:
        """Draw identifiable alignment marks."""
        margin = self.margin  # 1 cm
        page_width, page_height = self.page_width, self.page_height  # A4
        marker_size = self.marker_size  # 2 cm

        def _draw_aruco(c, x, y, size, marker_id=0, dictionary=cv2.aruco.DICT_6X6_250):
            """
            Draw ArUco marker
            """
            # Generate marker
            aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 8)

            # Get dimensions of marker
            marker_size = marker_img.shape[0]
            cell_size = size / marker_size

            # Draw ArUco square by square
            c.saveState()
            for i in range(marker_size):
                for j in range(marker_size):
                    if marker_img[i, j] == 0:
                        # flip y-axis since ReportLab uses bottom-left origin
                        pos_x = x + j * cell_size
                        pos_y = y + size - (i + 1) * cell_size
                        c.setFillColor(colors.black)
                        c.rect(pos_x, pos_y, cell_size, cell_size, fill=True, stroke=False)
            c.restoreState()

        # draw ArUco counterclock wise, from top left
        _draw_aruco(c, margin, page_height - margin - marker_size, marker_size, marker_id=0)
        _draw_aruco(c, margin, margin, marker_size, marker_id=1)
        _draw_aruco(c, page_width - margin - marker_size, margin, marker_size, marker_id=2)
        _draw_aruco(c, page_width - margin - marker_size, page_height - margin - marker_size, marker_size, marker_id=3)

        self.static_metrics.top_left = 0
        self.static_metrics.top_right = 3
        self.static_metrics.bottom_right = 2
        self.static_metrics.bottom_left = 1

    @staticmethod
    def _equal_bin_packing(num_item: int, num_bin: int, bin_cap: int) -> list[int]:
        """Find equal distribution of balance answer group across rows."""
        assert num_bin * bin_cap >= num_item, "parameters out of bound in bin packing function"
        # Base distribution and remainder
        base = num_item // num_bin
        remainder = num_item % num_bin
        # Create and return the distribution list
        return [base + 1 if i < remainder else base for i in range(num_bin)]
