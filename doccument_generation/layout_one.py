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
    A class to generate multiple-choice answer sheets with customizable layouts.
    """

    def __init__(self, output_dir="out/pdf", debug=True):
        """
        Initialize the answer sheet generator with default settings.

        Args:
            output_dir: Directory where PDF files will be saved
            debug: Whether to show debug bounding boxes and information
        """
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
        Generate an answer sheet with the specified number of questions and choices.

        Args:
            num_questions: Number of questions in the test
            choices_per_question: Number of choices per question (e.g., 4 for A, B, C, D)
            questions_per_group: For grouping questions visually
            sheet_id: Optional unique ID for the sheet, generated if not provided
            filename: Optional filename for the PDF, generated if not provided

        Returns:
            tuple: (filepath, sheet_id, metadata)
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

        Returns:
            float: Y-coordinate below the header
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
        Draw the form fields section (student ID, class, location).

        Args:
            c: ReportLab canvas object
            y_start: Starting Y-coordinate

        Returns:
            float: Y-coordinate below the form fields
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
        Draw a debug bounding box with optional label.

        Args:
            c: ReportLab canvas object
            x, y: Bottom-left coordinates of the box
            width, height: Dimensions of the box
            label: Optional label for the box
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
        Draw QR code with sheet ID, positioned on the right side.

        Args:
            c: ReportLab canvas object
            sheet_id: Unique identifier for this sheet
            y_start: Starting Y-coordinate
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

        Args:
            c: ReportLab canvas object
            num_questions: Total number of questions
            choices_per_question: Number of choice bubbles per question
            questions_per_group: Number of questions in each visual group
            y_start: Starting Y-coordinate

        Returns:
            float: Y-coordinate below the answer section
        """
        # Calculate available space
        available_height = y_start - (self.margin - self.marker_size)
        available_width = self.page_width - 2 * self.margin

        # Define layout dimensions
        choice_width = self.bubble_spacing
        question_height = 8 * mm
        question_number_width = 1 * cm
        answer_group_width = question_number_width + (choices_per_question * choice_width)

        # Calculate group layout
        groups_per_row = max(1, int(available_width / answer_group_width))
        horizontal_group_margin = (available_width - (groups_per_row * answer_group_width)) / max(1, groups_per_row - 1)
        group_height = (question_height * questions_per_group) + self.q_group_vertical_padding + self.group_header_height

        # Calculate total size required
        total_groups = (num_questions + questions_per_group - 1) // questions_per_group
        total_rows = (total_groups + groups_per_row - 1) // groups_per_row
        section_top_margin = 0.5 * cm
        total_height_needed = (total_rows * group_height) + section_top_margin

        # Scale down if not enough space
        if total_height_needed > available_height:
            available_height_for_groups = available_height - section_top_margin
            max_rows = int(available_height_for_groups / group_height)
            max_groups = max_rows * groups_per_row
            max_questions = max_groups * questions_per_group

            print(f"Warning: Not enough space for {num_questions} questions. Limiting to {max_questions} questions.")
            num_questions = max_questions

            # Recalculate layout
            total_groups = (num_questions + questions_per_group - 1) // questions_per_group
            total_rows = (total_groups + groups_per_row - 1) // groups_per_row
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

        # Draw answer groups
        for group_no in range(total_groups):
            # Calculate group position
            row = group_no // groups_per_row
            col = group_no % groups_per_row

            # Position group with proper margins
            if groups_per_row > 1:
                group_x = self.margin + (col * (answer_group_width + horizontal_group_margin))
            else:
                group_x = self.margin

            group_top_y = y_start - (row * group_height)

            # Calculate question range for this group
            first_q = (group_no * questions_per_group) + 1
            last_q = min((group_no + 1) * questions_per_group, num_questions)
            questions_in_group = last_q - first_q + 1

            # Calculate group height
            is_last_row = (row == total_rows - 1)
            actual_group_height = (questions_in_group * question_height) + self.group_header_height
            if not is_last_row:
                actual_group_height += self.q_group_vertical_padding

            # Draw group header
            c.setFont("Helvetica-Bold", 10)
            c.drawString(group_x, group_top_y, f"Questions {first_q} - {last_q}")

            # Draw group debug box
            self._draw_debug_box(c,
                                 group_x, group_top_y - actual_group_height,
                                 answer_group_width, actual_group_height,
                                 f"Group {group_no + 1}")

            # Draw questions in this group
            self._draw_question_group(c, group_x, group_top_y - self.group_header_height,
                                      first_q, last_q, question_height, question_number_width,
                                      choice_width, choices_per_question, choices)

        # Calculate final y position
        final_y = max(section_end_y - 0.5 * cm, self.margin)
        return final_y

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
        Draw a straight horizontal line across the page.

        Args:
            c: ReportLab canvas object
            y_position: Y-coordinate for the line
        """
        c.line(self.margin, y_position, self.page_width - self.margin, y_position)
