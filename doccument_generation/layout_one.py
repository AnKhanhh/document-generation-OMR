import os
import uuid
from datetime import datetime
from typing import Tuple, Dict, Any

from reportlab.graphics import renderPDF
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.pdfgen import canvas


class AnswerSheetGenerator:
    """
    Generator of multiple-choice answer sheets with dynamic layouts.
    """

    def __init__(self, output_dir="out/pdf", debug=True):
        """Initialize generator with default settings."""
        # Page dimensions
        self.page_width, self.page_height = A4

        # Document layout constants
        self.margin = 1 * cm
        self.header_margin = 1.5 * cm
        self.header_line_margin = 2.5 * cm
        self.marker_size = 0.5 * cm
        self.section_label_height = 0.5 * cm

        # Answer section layout constants
        self.bubble_radius = 3 * mm
        self.bubble_horizontal_space = 10 * mm
        self.answer_group_top_margin = 0.4 * cm
        self.answer_group_label_height = 0.5 * cm
        self.question_height = 0.8 * cm
        self.question_number_label_width = 1.2 * cm

        # Visual settings
        self.debug = debug
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def generate_answer_sheet(self,
                              num_questions: int = 30,
                              choices_per_question: int = 4,
                              questions_per_group: int = 5,
                              sheet_id: str = None,
                              filename: str = None) -> Tuple[str, str, Dict[str, Any]]:
        """Generate answer sheet. Returns filepath, sheet_id, custom metadata"""
        # Generate unique ID and filename if not provided
        sheet_id = sheet_id or str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"answer_sheet_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)

        # Create canvas and set metadata
        c = canvas.Canvas(filepath, pagesize=A4)
        c.setTitle(datetime.now().strftime("%Y/%m/%d_%H:%M"))

        # Sheet metadata
        metadata = {
            "sheet_id": sheet_id,
            "num_questions": num_questions,
            "choices_per_question": choices_per_question,
            "created_at": datetime.now().isoformat(),
            "filename": filename
        }

        # Draw all sheet components in sequence
        self._draw_alignment_markers(c)
        header_y = self._draw_header(c)
        form_region_y = self._draw_form_fields(c, header_y)
        self._draw_barcode(c, sheet_id, header_y)
        self._draw_answer_section(c, num_questions, choices_per_question, questions_per_group, form_region_y)

        # Save the canvas
        c.save()

        return filepath, sheet_id, metadata

    def _draw_header(self, c: canvas) -> float:
        """Draw the header of the answer sheet."""
        # Main title
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(self.page_width / 2, self.page_height - self.header_margin, "ANSWER SHEET")

        # Instructions or debug message
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
        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.margin, y_start, "INFORMATION FIELDS")

        y_start -= 1 * cm

        # Define field height and spacing
        field_height = 0.8 * cm
        field_label_width = 3 * cm
        field_width = 10 * cm
        field_spacing = 1.25 * cm

        # Common field drawing function
        def draw_field(label, y_pos):
            c.setFont("Helvetica-Bold", 10)
            c.drawString(self.margin, y_pos, label)
            c.rect(self.margin + field_label_width, y_pos - 2 * mm, field_width, field_height)
            return y_pos - field_spacing

        # Draw each field
        y_start = draw_field("Student ID:", y_start)
        y_start = draw_field("Class:", y_start)
        y_start = draw_field("Location:", y_start)

        # Draw horizontal line
        c.setStrokeColor(colors.black)
        self._draw_horizontal_line(c, y_start)

        return y_start

    def _draw_debug_box(self, c: canvas, x: float, y: float, width: float, height: float, label: str = None) -> None:
        """Draw a debug bounding box, label optional."""
        assert self.debug, "debug function evoked unintentionally"

        c.saveState()
        c.setStrokeColor(colors.red)
        c.setLineWidth(0.5)
        c.rect(x, y, width, height, stroke=1, fill=0)
        if label:
            c.setFillColor(colors.red)
            c.setFont("Helvetica", 8)
            c.drawString(x, y + height + 2 * mm, label)
        c.restoreState()

    def _draw_barcode(self, c: canvas, sheet_id: str, y_start: float) -> None:
        """Draw sheet_id QR code, right-aligned."""
        # Define QR code basic parameters
        qr_size = 4 * cm
        qr_x = self.page_width - self.margin - qr_size

        # Create QR code
        qr_code = qr.QrCodeWidget(sheet_id)
        bounds = qr_code.getBounds()
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        # Draw QR code
        d = Drawing(qr_size, qr_size, transform=[qr_size / width, 0, 0, qr_size / height, 0, 0])
        d.add(qr_code)
        renderPDF.draw(d, c, qr_x, y_start - qr_size)

        # When debug, draw label and ID string
        if self.debug:
            c.saveState()
            c.setFillColor(colors.red)
            c.setFont("Helvetica-Bold", 12)
            c.drawCentredString(qr_x + qr_size / 2, y_start, "SHEET ID")
            c.rotate(90)
            c.setFont("Helvetica", 8)
            c.drawString(y_start - 5 * cm, - self.page_width + 5 * mm, f"ID: {sheet_id}")
            c.restoreState()

    def _draw_answer_section(self, c: canvas,
                             num_questions: int, choices_per_question: int, questions_per_group: int,
                             y_start: float) -> float:
        """
        Draw the answer bubbles section with proper spacing and layout management.
        Some values are defined in build time, but can be overwritten at run time dynamically
        """
        # 1.Define basic dimensions for answer section
        available_height = y_start - (self.margin + self.marker_size)
        available_width = self.page_width - 2 * self.margin
        answer_section_label_height = self.section_label_height
        choice_arr = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # 2.Initialize subsections (answer groups) parameters
        choice_width = self.bubble_horizontal_space
        question_number_width = self.question_number_label_width
        group_width = question_number_width + (choices_per_question * choice_width)
        group_height = (self.question_height * questions_per_group) + self.answer_group_label_height
        group_y_gap = self.answer_group_top_margin

        # 3.Calculate layout limits, recalibrate parameters to fit
        # Calibrate width
        if (max_groups_allowed_per_row := int(available_width / group_width)) < 1:
            choices_per_question = int((available_width - question_number_width) / choice_width)
            print(f"Warning: insufficient width, limiting to {choices_per_question} choices per questions")

            group_width = question_number_width + (choices_per_question * choice_width)
            max_groups_allowed_per_row = int(available_width / group_width)
            assert max_groups_allowed_per_row == 1, "answer section width calibration failed"

        # Number of groups needed to hold all questions
        num_group = (num_questions + questions_per_group - 1) // questions_per_group
        # Number of rows needed to hold all groups
        num_group_row = (num_group + max_groups_allowed_per_row - 1) // max_groups_allowed_per_row

        # Calibrate height
        # noinspection PyUnusedLocal
        # If not enough vertical space, fit as many answer groups as possible
        if (total_height_needed := (num_group_row * (group_height + group_y_gap)) + answer_section_label_height) > available_height:
            # Recalculate metrics based on limitations
            num_group_row = int((available_height - answer_section_label_height) / (group_height + group_y_gap))
            total_height_needed = (num_group_row * (group_height + group_y_gap)) + answer_section_label_height
            num_group = num_group_row * max_groups_allowed_per_row
            num_questions = num_group * questions_per_group
            group_distribution_on_rows: list[int] = [max_groups_allowed_per_row for _ in range(num_group_row)]
            print(f"Warning: insufficient length, limiting to {num_questions} questions.")

            total_height_needed = (num_group_row * (group_height + group_y_gap)) + answer_section_label_height
            assert total_height_needed <= available_height, "answer section height calibration failed"

        # If enough vertical space, try to spread groups evenly across rows
        else:
            max_rows_allowed = int((available_height - answer_section_label_height) / (group_height + group_y_gap))
            group_distribution_on_rows: list = self._equal_bin_packing(num_group, max_rows_allowed, max_groups_allowed_per_row)
            num_group_row = len(group_distribution_on_rows)
            total_height_needed = (num_group_row * (group_height + group_y_gap)) + answer_section_label_height

        # Calculate section boundaries
        section_end_y = max(y_start - total_height_needed, self.margin + self.marker_size)
        section_height = y_start - section_end_y

        # Answer section debug box
        if self.debug:
            self._draw_debug_box(c,
                                 self.margin, section_end_y,
                                 available_width, section_height,
                                 "Answer Section Box")

        # Draw section label
        y_start -= answer_section_label_height
        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.margin, y_start, "ANSWER SECTION")

        # Answer group horizontal margin is static across rows, for better OMR
        if (groups_on_row := group_distribution_on_rows[0]) > 1:
            group_x_gap = (available_width - (groups_on_row * group_width)) / (groups_on_row - 1)
        else:
            group_x_gap = 0

        # 4.Draw answer groups, row by row
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

                group_x = self.margin + (col * (group_width + group_x_gap))

                # Calculate question range for this group
                first_q = (current_group * questions_per_group) + 1
                last_q = min((current_group + 1) * questions_per_group, num_questions)

                if self.debug:
                    self._draw_debug_box(c, group_x, row_top_y - group_height, group_width, group_height,
                                         f"Group {current_group + 1} Box")

                # Draw group header, offset to add space at bottom
                c.setFont("Helvetica-Bold", 10)
                c.drawString(group_x, row_top_y - self.answer_group_label_height + 1 * mm,
                             f"Questions {first_q} - {last_q}")

                # Draw questions in group
                self._draw_question_group(c, group_x, row_top_y - self.answer_group_label_height,
                                          first_q, last_q, self.question_height, question_number_width,
                                          choice_width, choices_per_question, choice_arr)

                # Move to next group
                current_group += 1

            y_start -= group_height

        # Calculate final y position
        final_y = max(section_end_y - 0.5 * cm, self.margin)
        return final_y

    def _draw_question_group(self, c: canvas, x: float, y: float,
                             first_q: int, last_q: int,
                             question_height: float, question_number_width: float,
                             choice_width: float, choices_per_question: int, choices: str) -> None:
        """
        Draw a group of questions with answer bubbles.
        Reportlab bubbles are drawn from center, code needs to accommodate that offset.
        """
        answer_y = y

        for q_no in range(first_q, last_q + 1):
            answer_y -= question_height / 2
            # 1mm offset to align number with bubble
            c.setFont("Helvetica", 10)
            c.drawString(x, answer_y - 1 * mm, f"{q_no}.")

            if self.debug:
                self._draw_debug_box(c, x, answer_y - question_height / 2,
                                     question_number_width - choice_width / 2, question_height)

            # Draw bubbles
            for choice_no in range(choices_per_question):
                # Calculate bubble position
                choice_x = x + question_number_width + (choice_no * choice_width)
                # Draw bubble and letter
                c.circle(choice_x, answer_y, self.bubble_radius, stroke=1, fill=0)
                letter = choices[choice_no]
                self._draw_centered_letter(c, letter, choice_x, answer_y)

                if self.debug:
                    self._draw_debug_box(c, choice_x - choice_width / 2, answer_y - question_height / 2,
                                         choice_width, question_height)

            answer_y -= question_height / 2

    def _draw_horizontal_line(self, c: canvas, y_position: float) -> None:
        """Draw a straight line, account for margin."""
        c.line(self.margin, y_position, self.page_width - self.margin, y_position)

    @staticmethod
    def draw_horizontal_dashed_line(self, c: canvas, x_start, x_end, y: float, width=1, color=colors.red):
        """Draw a horizontal dashed line"""
        c.saveState()

        c.setLineWidth(width)
        c.setStrokeColor(color)
        c.setDash([4 * mm, 4 * mm])
        c.line(x_start, y, x_end, y)

        c.restoreState()

    @staticmethod
    def _draw_centered_letter(c: canvas, letter: str, x: float, y: float) -> None:
        """Draw a letter centered in a bubble."""
        c.setFont("Helvetica", 8)
        # Center the letter in the bubble
        letter_width = c.stringWidth(letter, "Helvetica", 8)
        letter_x = x - (letter_width / 2)
        letter_y = y - (c._leading / 4)  # Adjust for vertical centering

        c.drawString(letter_x, letter_y, letter)

    def _draw_alignment_markers(self, c: canvas) -> None:
        """Draw alignment markers in the corners for easier scanning."""
        offset = self.margin + self.marker_size

        # Draw markers in each corner: top-left, top-right, bot-left, bot-right
        c.rect(self.margin, self.page_height - offset, self.marker_size, self.marker_size, fill=1)
        c.rect(self.page_width - offset, self.page_height - offset, self.marker_size, self.marker_size, fill=1)
        c.rect(self.margin, self.margin, self.marker_size, self.marker_size, fill=1)
        c.rect(self.page_width - offset, self.margin, self.marker_size, self.marker_size, fill=1)

    @staticmethod
    def _equal_bin_packing(num_item: int, num_bin: int, bin_cap: int) -> list[int]:
        """Find equal distribution of balance answer group across rows."""
        assert num_bin * bin_cap >= num_item, "parameters out of bound in bin packing function"
        # Base distribution and remainder
        base = num_item // num_bin
        remainder = num_item % num_bin
        # Create and return the distribution list
        return [base + 1 if i < remainder else base for i in range(num_bin)]
