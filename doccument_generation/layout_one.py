import os
import uuid
from datetime import datetime

from reportlab.graphics import renderPDF
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.pdfgen import canvas


class AnswerSheetGenerator:
    def __init__(self, output_dir="out/pdf", debug=True):
        self.page_width, self.page_height = A4
        self.margin = 1 * cm
        self.bubble_radius = 3 * mm
        self.bubble_spacing = 10 * mm
        self.output_dir = output_dir
        self.q_group_vertical_padding = 0.7 * cm
        self.marker_size = 0.5 * cm
        self.group_header_height = 0.5 * cm  # Height for "Questions X-Y" text

        # Debug mode for showing bounding boxes
        self.debug = debug

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_answer_sheet(self, num_questions=30, choices_per_question=4, questions_per_group=5,
                              sheet_id=None, filename=None):
        """
        Generate an answer sheet with the specified number of questions and choices.

        Args:
            num_questions: Number of questions in the test
            choices_per_question: Number of choices per question (e.g., 4 for A, B, C, D)
            questions_per_group: For grouping questions visually
            sheet_id: Optional unique ID for the sheet, generated if not provided
            filename: Optional filename for the PDF, generated if not provided

        Returns:
            tuple: (filename, sheet_id, metadata)
        """
        # Generate unique ID if not provided
        if sheet_id is None:
            sheet_id = str(uuid.uuid4())

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"answer_sheet_{timestamp}.pdf"

        filepath = os.path.join(self.output_dir, filename)

        # Create canvas
        c = canvas.Canvas(filepath, pagesize=A4)
        c.setTitle(datetime.now().strftime("%Y/%m/%d_%H:%M"))

        # Metadata for the answer sheet
        metadata = {
            "sheet_id": sheet_id,
            "num_questions": num_questions,
            "choices_per_question": choices_per_question,
            "created_at": datetime.now().isoformat(),
            "filename": filename
        }

        # Draw alignment markers
        self._draw_alignment_markers(c, self.marker_size)

        # Draw the header and title
        header_y = self._draw_header(c)

        # Draw form fields (student ID, class, location) and QR code
        form_region_y = self._draw_form_fields(c, header_y)
        self._draw_barcode(c, sheet_id, header_y)  # QR code at same y-level

        # Calculate available space and draw answer section
        answer_region_y = self._draw_answer_section(c, num_questions, choices_per_question, questions_per_group, form_region_y)

        # Save the canvas
        c.save()

        return filepath, sheet_id, metadata

    def _draw_header(self, c: canvas) -> float:
        """Draw the header of the answer sheet."""
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(self.page_width / 2, self.page_height - 1.5 * cm, "ANSWER SHEET")

        c.setFont("Helvetica", 10)
        if self.debug:
            c.saveState()
            c.setFillColor(colors.red)
            c.drawCentredString(self.page_width / 2, self.page_height - 2 * cm, "Debug mode")
            c.restoreState()
        else:
            c.drawCentredString(self.page_width / 2, self.page_height - 2 * cm, "Fill the bubbles completely using a dark pen or pencil")

        # Draw horizontal line
        c.setStrokeColor(colors.black)
        self._draw_horizontal_line(c, self.page_height - 2.5 * cm)

        return self.page_height - 3 * cm

    def _draw_form_fields(self, c: canvas, y_start: float) -> float:
        """Draw the form fields section (student ID, class, location)."""
        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.margin, y_start, "INFORMATION FIELDS")
        y_start -= 1 * cm

        # Student ID Field
        c.setFont("Helvetica-Bold", 10)
        c.drawString(self.margin, y_start, "Student ID:")
        c.rect(self.margin + 3 * cm, y_start - 2 * mm, 10 * cm, 0.8 * cm)

        # Class Field
        y_start -= 1.25 * cm
        c.drawString(self.margin, y_start, "Class:")
        c.rect(self.margin + 3 * cm, y_start - 2 * mm, 10 * cm, 0.8 * cm)

        # Location Field
        y_start -= 1.25 * cm
        c.drawString(self.margin, y_start, "Location:")
        c.rect(self.margin + 3 * cm, y_start - 2 * mm, 10 * cm, 0.8 * cm)

        # Draw horizontal line
        y_start -= 1.25 * cm
        c.setStrokeColor(colors.black)
        self._draw_horizontal_line(c, y_start)

        return y_start - 0.5 * cm

    def _draw_debug_box(self, c, x, y, width, height, label=None):
        """Draw a debug bounding box."""
        if self.debug:
            # Save current graphics state
            c.saveState()

            # Set thin red stroke for debug boxes
            c.setStrokeColor(colors.red)
            c.setLineWidth(0.5)
            c.rect(x, y, width, height, stroke=1, fill=0)

            # Add label if provided
            if label:
                c.setFillColor(colors.red)
                c.setFont("Helvetica", 8)
                c.drawString(x, y + height + 2 * mm, label)

            # Restore graphics state
            c.restoreState()

    def _draw_barcode(self, c, sheet_id, y_start):
        """Draw QR code with sheet ID, positioned on the right side."""
        # Calculate positions for right-aligned QR code
        qr_size = 4 * cm
        qr_x = self.page_width - self.margin - qr_size  # Right-aligned

        # Create QR code
        qr_code = qr.QrCodeWidget(sheet_id)
        bounds = qr_code.getBounds()
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        # Create drawing for QR code
        d = Drawing(qr_size, qr_size, transform=[qr_size / width, 0, 0, qr_size / height, 0, 0])
        d.add(qr_code)

        if self.debug:
            c.saveState()
            c.setFillColor(colors.red)
            c.setFont("Helvetica-Bold", 12)
            c.drawCentredString(qr_x + qr_size / 2, y_start, "SHEET ID")
            c.restoreState()

        # Position and render QR code
        renderPDF.draw(d, c, qr_x, y_start - qr_size)

        # Add sheet ID as vertical text when in debug mode
        if self.debug:
            c.saveState()
            # Draw in margin
            c.rotate(90)
            c.setFont("Helvetica", 8)
            c.setFillColor(colors.red)
            c.drawString(y_start - 5 * cm, - self.page_width + 5 * mm, f"ID: {sheet_id}")
            c.restoreState()

    def _draw_answer_section(self, c: canvas,
                             num_questions: int, choices_per_question: int, questions_per_group: int,
                             y_start: float) -> float:
        """Draw the answer bubbles section with proper spacing and layout management."""
        # Available space for the answer section
        available_height = y_start - (self.margin - self.marker_size)
        available_width = self.page_width - 2 * self.margin

        # Define space needed for each choice bubble and letter
        choice_width = self.bubble_spacing
        # Define height needed for each question
        question_height = 8 * mm

        # Calculate width needed for a single answer group (question number + all choices)
        question_number_width = 1 * cm
        answer_group_width = question_number_width + (choices_per_question * choice_width)

        # Determine how many groups can fit horizontally in a row
        groups_per_row = max(1, int(available_width / answer_group_width))

        # Calculate horizontal group margin to ensure proper spacing between groups
        horizontal_group_margin = (available_width - (groups_per_row * answer_group_width)) / max(1, groups_per_row - 1)

        # Calculate height needed for each group with vertical padding and header
        # Always include group header height whether in debug mode or not
        group_header_height = self.group_header_height
        group_height = (question_height * questions_per_group) + self.q_group_vertical_padding + group_header_height

        # Calculate total number of groups
        total_groups = (num_questions + questions_per_group - 1) // questions_per_group

        # Calculate number of rows needed
        total_rows = (total_groups + groups_per_row - 1) // groups_per_row

        # Calculate total height needed
        section_top_margin = 0.5 * cm
        total_height_needed = (total_rows * group_height) + section_top_margin

        # Check if we have enough space, adjust if necessary
        if total_height_needed > available_height:
            # Calculate how many questions we can fit
            available_height_for_groups = available_height - section_top_margin
            max_rows = int(available_height_for_groups / group_height)
            max_groups = max_rows * groups_per_row
            max_questions = max_groups * questions_per_group

            print(f"Warning: Not enough space for {num_questions} questions. Limiting to {max_questions} questions.")
            num_questions = max_questions

            # Recalculate total groups and height needed
            total_groups = (num_questions + questions_per_group - 1) // questions_per_group
            total_rows = (total_groups + groups_per_row - 1) // groups_per_row
            total_height_needed = (total_rows * group_height) + section_top_margin

        # section title
        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.margin, y_start, "ANSWER SECTION")

        # Answer section y-coordinate starts
        y_start -= section_top_margin

        choices = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        section_end_y = y_start - total_height_needed

        # Ensure answer section doesn't go below margin
        section_end_y = max(section_end_y, self.margin + self.marker_size)
        section_height = y_start - section_end_y

        # Answer section debug box
        self._draw_debug_box(c,
                             self.margin - 0.5 * cm, section_end_y,
                             available_width + 1 * cm, section_height,
                             "(Debug) Answer Section")

        # Answer section top padding
        y_start -= 0.5 * cm

        # Draw answer grid
        for group_no in range(total_groups):
            # Calculate group position (row and column)
            row = group_no // groups_per_row
            col = group_no % groups_per_row

            # Calculate group boundaries with proper margins between groups
            if groups_per_row > 1:
                # Use horizontal margin to position groups
                group_x = self.margin + (col * (answer_group_width + horizontal_group_margin))
            else:
                # Center single group
                group_x = self.margin

            group_top_y = y_start - (row * group_height)

            # Calculate first and last question in this group
            first_q = (group_no * questions_per_group) + 1
            last_q = min((group_no + 1) * questions_per_group, num_questions)

            # Calculate actual questions in this group
            questions_in_group = last_q - first_q + 1

            # Calculate actual height for this group
            is_last_row = (row == total_rows - 1)
            actual_group_height = (questions_in_group * question_height) + group_header_height
            if not is_last_row:
                actual_group_height += self.q_group_vertical_padding

            # Draw group header
            c.setFont("Helvetica-Bold", 10)
            c.drawString(group_x, group_top_y, f"Questions {first_q} - {last_q}")

            # Debug - draw group box
            self._draw_debug_box(c,
                                 group_x, group_top_y - actual_group_height,
                                 answer_group_width, actual_group_height,
                                 f"Group {group_no + 1}")

            # Draw questions in this group, starting below the header
            question_y = group_top_y - group_header_height

            for q_no in range(first_q, last_q + 1):
                # Draw question number
                c.setFont("Helvetica", 10)
                c.drawString(group_x, question_y - 2 * mm, f"{q_no}.")

                # Draw bubbles for each choice horizontally
                for choice_no in range(choices_per_question):
                    # Position for this choice
                    choice_x = group_x + question_number_width + (choice_no * choice_width)

                    # Draw the bubble
                    c.circle(choice_x, question_y, self.bubble_radius, stroke=1, fill=0)

                    # Draw the choice letter inside the bubble
                    c.setFont("Helvetica", 8)  # Smaller font for inside the bubble
                    letter = choices[choice_no]
                    # Center the letter in the bubble
                    letter_width = c.stringWidth(letter, "Helvetica", 8)
                    letter_x = choice_x - (letter_width / 2)
                    letter_y = question_y - (c._leading / 4)  # Adjust for vertical centering
                    c.drawString(letter_x, letter_y, letter)

                # Move to next question
                question_y -= question_height

        # Calculate final y position, ensuring it doesn't go below margin
        final_y = max(section_end_y - 0.5 * cm, self.margin)
        return final_y

    def _draw_alignment_markers(self, c: canvas, marker_size: float) -> None:
        """Draw alignment markers in the corners for easier scanning."""
        offset = self.margin + marker_size

        # Top-left
        c.rect(self.margin, self.page_height - offset, marker_size, marker_size, fill=1)
        # Top-right
        c.rect(self.page_width - offset, self.page_height - offset, marker_size, marker_size, fill=1)
        # Bottom-left
        c.rect(self.margin, self.margin, marker_size, marker_size, fill=1)
        # Bottom-right
        c.rect(self.page_width - offset, self.margin, marker_size, marker_size, fill=1)

    def _draw_horizontal_line(self, c: canvas, y_position: float) -> None:
        """Draw a straight horizontal line"""
        c.line(self.margin, y_position, self.page_width - self.margin, y_position)
