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

        # Layout constants
        self.margin = 1 * cm
        self.bubble_radius = 3 * mm
        self.bubble_spacing = 10 * mm
        self.q_group_vertical_padding = 0.7 * cm
        self.marker_size = 0.5 * cm
        self.group_header_height = 0.5 * cm

        # Section spacing constants
        self.header_margin = 1.5 * cm
        self.header_line_margin = 2.5 * cm
        self.section_spacing = 0.5 * cm

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
        """
        Generate answer sheet. Returns filepath, sheet_id, custom metadata
        """
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
        """
        Draw the header of the answer sheet.
        """
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

        return self.page_height - 3 * cm

    def _draw_form_fields(self, c: canvas, y_start: float) -> float:
        """
        Draw the information fields
        """
        # Section header
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

        return y_start - self.section_spacing

    def _draw_debug_box(self, c: canvas, x: float, y: float, width: float, height: float, label: str = None) -> None:
        """
        Draw a debug bounding box, label optional.
        """
        if self.debug:
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
        """
        Draw sheet_id QR code, right-aligned.
        """
        # QR code dimensions and position
        qr_size = 4 * cm
        qr_x = self.page_width - self.margin - qr_size

        # Create QR code
        qr_code = qr.QrCodeWidget(sheet_id)
        bounds = qr_code.getBounds()
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        # Create drawing for QR code
        d = Drawing(qr_size, qr_size, transform=[qr_size / width, 0, 0, qr_size / height, 0, 0])
        d.add(qr_code)

        # Debug label
        if self.debug:
            c.saveState()
            c.setFillColor(colors.red)
            c.setFont("Helvetica-Bold", 12)
            c.drawCentredString(qr_x + qr_size / 2, y_start, "SHEET ID")
            c.restoreState()

        # Render QR code
        renderPDF.draw(d, c, qr_x, y_start - qr_size)

        # Add sheet ID as vertical text when in debug mode
        if self.debug:
            c.saveState()
            c.rotate(90)
            c.setFont("Helvetica", 8)
            c.setFillColor(colors.red)
            c.drawString(y_start - 5 * cm, - self.page_width + 5 * mm, f"ID: {sheet_id}")
            c.restoreState()

    def _draw_answer_section(self, c: canvas,
                             num_questions: int, choices_per_question: int, questions_per_group: int,
                             y_start: float) -> float:
        """
        Draw the answer bubbles section with proper spacing and layout management.
        """
        # Calculate available space
        available_height = y_start - (self.margin - self.marker_size)
        available_width = self.page_width - 2 * self.margin

        # Define layout dimensions
        choice_width = self.bubble_spacing
        question_height = 8 * mm
        question_number_width = 1 * cm
        answer_group_width = question_number_width + (choices_per_question * choice_width)

        # Calculate maximum groups per row based on available width
        max_groups_per_row = max(1, int(available_width / answer_group_width))
        group_height = (question_height * questions_per_group) + self.q_group_vertical_padding + self.group_header_height

        # Calculate total groups needed
        total_groups = (num_questions + questions_per_group - 1) // questions_per_group

        # Calculate minimum number of rows needed
        min_rows_needed = (total_groups + max_groups_per_row - 1) // max_groups_per_row

        # Calculate balanced distribution of groups per row
        groups_per_row = self._calculate_balanced_distribution(total_groups, min_rows_needed)

        # Calculate total height needed based on the number of rows
        total_rows = len(groups_per_row)
        section_top_margin = 0.5 * cm
        total_height_needed = (total_rows * group_height) + section_top_margin

        # Scale down if not enough space
        if total_height_needed > available_height:
            available_height_for_groups = available_height - section_top_margin
            max_rows = int(available_height_for_groups / group_height)

            if max_rows < min_rows_needed:
                # Not enough vertical space, need to reduce number of questions
                # Calculate max groups that can fit
                max_groups = 0
                for i in range(max_rows):
                    max_groups += groups_per_row[i] if i < len(groups_per_row) else 0

                max_questions = max_groups * questions_per_group
                print(f"Warning: Not enough space for {num_questions} questions. Limiting to {max_questions} questions.")
                num_questions = max_questions

                # Recalculate distribution
                total_groups = (num_questions + questions_per_group - 1) // questions_per_group
                groups_per_row = self._calculate_balanced_distribution(total_groups, max_rows)
                total_rows = len(groups_per_row)
                total_height_needed = (total_rows * group_height) + section_top_margin

        # Draw section header
        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.margin, y_start, "ANSWER SECTION")

        # Set Y-coordinate for section content
        y_start -= section_top_margin
        choices = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # Calculate section boundaries
        section_end_y = y_start - total_height_needed
        section_end_y = max(section_end_y, self.margin + self.marker_size)
        section_height = y_start - section_end_y

        # Draw section debug box
        self._draw_debug_box(c,
                             self.margin - 0.5 * cm, section_end_y,
                             available_width + 1 * cm, section_height,
                             "(Debug) Answer Section")

        # Apply section top padding
        y_start -= 0.5 * cm

        # Track the current group being drawn
        current_group = 0

        # Draw answer groups row by row
        for row in range(total_rows):
            # Calculate groups in this row and their horizontal spacing
            groups_in_this_row = groups_per_row[row]

            # Evenly distribute groups across available width
            if groups_in_this_row > 1:
                horizontal_group_margin = (available_width - (groups_in_this_row * answer_group_width)) / (groups_in_this_row - 1)
            else:
                # Center a single group
                horizontal_group_margin = 0

            # Set Y-coordinate for this row
            row_top_y = y_start - (row * group_height)

            # Draw all groups in this row
            for col in range(groups_in_this_row):
                if current_group >= total_groups:
                    break

                # Calculate horizontal position (centered if single group)
                if groups_in_this_row > 1:
                    group_x = self.margin + (col * (answer_group_width + horizontal_group_margin))
                else:
                    # Center single group
                    group_x = self.margin + (available_width - answer_group_width) / 2

                # Calculate question range for this group
                first_q = (current_group * questions_per_group) + 1
                last_q = min((current_group + 1) * questions_per_group, num_questions)
                questions_in_group = last_q - first_q + 1

                # Calculate group height
                is_last_row = (row == total_rows - 1)
                actual_group_height = (questions_in_group * question_height) + self.group_header_height
                if not is_last_row:
                    actual_group_height += self.q_group_vertical_padding

                # Draw group header
                c.setFont("Helvetica-Bold", 10)
                c.drawString(group_x, row_top_y, f"Questions {first_q} - {last_q}")

                # Draw group debug box
                self._draw_debug_box(c,
                                     group_x, row_top_y - actual_group_height,
                                     answer_group_width, actual_group_height,
                                     f"Group {current_group + 1}")

                # Draw questions in this group
                self._draw_question_group(c, group_x, row_top_y - self.group_header_height,
                                          first_q, last_q, question_height, question_number_width,
                                          choice_width, choices_per_question, choices)

                # Move to next group
                current_group += 1

        # Calculate final y position
        final_y = max(section_end_y - 0.5 * cm, self.margin)
        return final_y

    def _calculate_balanced_distribution(self, total_items: int, num_rows: int) -> list:
        """
        Calculate a balanced distribution of items across rows.

        Args:
            total_items: Total number of items to distribute
            num_rows: Number of rows to distribute across

        Returns:
            list: Number of items per row
        """
        # Base distribution - equal items per row
        items_per_row = [total_items // num_rows] * num_rows

        # Distribute remainder from top to bottom
        remainder = total_items % num_rows
        for i in range(remainder):
            items_per_row[i] += 1

        # Remove any empty rows
        return [count for count in items_per_row if count > 0]

    def _draw_question_group(self, c: canvas, x: float, y: float,
                             first_q: int, last_q: int,
                             question_height: float, question_number_width: float,
                             choice_width: float, choices_per_question: int, choices: str) -> None:
        """
        Draw a group of questions with answer bubbles.

        Args:
            c: ReportLab canvas object
            x, y: Top-left position of the question group
            first_q, last_q: Question number range
            question_height: Vertical space for each question
            question_number_width: Width reserved for question number
            choice_width: Width for each choice bubble
            choices_per_question: Number of choices per question
            choices: String of available choice letters
        """
        question_y = y

        for q_no in range(first_q, last_q + 1):
            # Draw question number
            c.setFont("Helvetica", 10)
            c.drawString(x, question_y - 2 * mm, f"{q_no}.")

            # Draw answer choices
            for choice_no in range(choices_per_question):
                # Calculate bubble position
                choice_x = x + question_number_width + (choice_no * choice_width)

                # Draw the bubble
                c.circle(choice_x, question_y, self.bubble_radius, stroke=1, fill=0)

                # Draw the choice letter inside the bubble
                letter = choices[choice_no]
                self._draw_centered_letter(c, letter, choice_x, question_y)

            # Move to next question
            question_y -= question_height

    def _draw_centered_letter(self, c: canvas, letter: str, x: float, y: float) -> None:
        """
        Draw a letter centered in a bubble.

        Args:
            c: ReportLab canvas object
            letter: The letter to draw
            x, y: Center coordinates of the bubble
        """
        c.setFont("Helvetica", 8)  # Smaller font for inside the bubble

        # Center the letter in the bubble
        letter_width = c.stringWidth(letter, "Helvetica", 8)
        letter_x = x - (letter_width / 2)
        letter_y = y - (c._leading / 4)  # Adjust for vertical centering

        c.drawString(letter_x, letter_y, letter)

    def _draw_alignment_markers(self, c: canvas) -> None:
        """
        Draw alignment markers in the corners for easier scanning.

        Args:
            c: ReportLab canvas object
        """
        offset = self.margin + self.marker_size

        # Draw markers in each corner
        c.rect(self.margin, self.page_height - offset, self.marker_size, self.marker_size, fill=1)
        c.rect(self.page_width - offset, self.page_height - offset, self.marker_size, self.marker_size, fill=1)
        c.rect(self.margin, self.margin, self.marker_size, self.marker_size, fill=1)
        c.rect(self.page_width - offset, self.margin, self.marker_size, self.marker_size, fill=1)

    def _draw_horizontal_line(self, c: canvas, y_position: float) -> None:
        """
        Draw a straight line, account for margin.
        """
        c.line(self.margin, y_position, self.page_width - self.margin, y_position)
