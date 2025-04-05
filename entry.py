from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.pdfgen import canvas
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing
from reportlab.graphics import renderPDF
import uuid
import json
import os
from datetime import datetime


class AnswerSheetGenerator:
    def __init__(self, output_dir="out/pdf", debug=True):
        self.page_width, self.page_height = A4
        self.margin = 1 * cm
        self.bubble_radius = 3 * mm
        self.bubble_spacing = 10 * mm
        self.output_dir = output_dir

        # Debug mode for showing bounding boxes
        self.debug = debug

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_answer_sheet(self, num_questions=30, choices_per_question=4,
                              sheet_id=None, filename=None):
        """
        Generate an answer sheet with the specified number of questions and choices.

        Args:
            num_questions: Number of questions in the test
            choices_per_question: Number of choices per question (e.g., 4 for A, B, C, D)
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

        # Metadata for the answer sheet
        metadata = {
            "sheet_id": sheet_id,
            "num_questions": num_questions,
            "choices_per_question": choices_per_question,
            "created_at": datetime.now().isoformat(),
            "filename": filename
        }

        # Draw alignment markers
        self._draw_alignment_markers(c)

        # Draw the header and title
        header_y = self._draw_header(c)

        # Draw form fields (student ID, class, location) and QR code
        form_region_y = self._draw_form_fields(c, header_y)
        self._draw_barcode(c, sheet_id, header_y)  # QR code at same y-level

        # Calculate available space and draw answer section
        answer_region_y = self._draw_answer_section(c, num_questions, choices_per_question, form_region_y)

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

        return self.page_height - 3.5 * cm

    def _draw_form_fields(self, c: canvas, y_start):
        """Draw the form fields section (student ID, class, location)."""
        # Student ID Field
        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.margin, y_start, "Student ID:")
        c.rect(self.margin + 3 * cm, y_start - 2 * mm, 10 * cm, 0.8 * cm)

        # Class Field
        y_start -= 2 * cm
        c.drawString(self.margin, y_start, "Class:")
        c.rect(self.margin + 3 * cm, y_start - 2 * mm, 10 * cm, 0.8 * cm)

        # Location Field
        y_start -= 2 * cm
        c.drawString(self.margin, y_start, "Location:")
        c.rect(self.margin + 3 * cm, y_start - 2 * mm, 10 * cm, 0.8 * cm)

        # Draw horizontal line
        y_start -= 1.5 * cm
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
        renderPDF.draw(d, c, qr_x, y_start - qr_size - 0.5 * cm)

        # Add sheet ID as vertical text when in debug mode
        if self.debug:
            c.saveState()
            # Draw in margin
            c.rotate(90)
            c.setFont("Helvetica", 8)
            c.setFillColor(colors.red)
            c.drawString(y_start - 5 * cm, - self.page_width + 5 * mm, f"ID: {sheet_id}")

            c.restoreState()

    # No need to return new y position as this is side-by-side with form

    def _draw_answer_section(self, c: canvas, num_questions, choices_per_question, y_start) -> float:
        """Draw the answer bubbles section with proper spacing and layout management."""
        # Available height for the answer section
        available_height = y_start - self.margin

        # Calculate the available width for the answer section
        available_width = self.page_width - 2 * self.margin

        # Calculate space needed for each choice bubble and letter
        choice_width = self.bubble_spacing * 1.5

        # Calculate width needed for a single answer group (question number + all choices)
        question_number_width = 1 * cm
        answer_group_width = question_number_width + (choices_per_question * choice_width)

        # Determine how many groups can fit horizontally in a row
        groups_per_row = max(1, int(available_width / answer_group_width))

        # Define how many questions per group (for organizing visually)
        questions_per_group = 5

        # Calculate height needed for each question
        question_height = 1.2 * cm

        # Calculate height needed for each group
        group_height = question_height * questions_per_group

        # Calculate total number of groups
        total_groups = (num_questions + questions_per_group - 1) // questions_per_group

        # Calculate number of rows needed
        total_rows = (total_groups + groups_per_row - 1) // groups_per_row

        # Calculate total height needed
        total_height_needed = (total_rows * group_height) + 2 * cm  # Extra space for headers

        # Check if we have enough space, adjust if necessary
        if total_height_needed > available_height:
            # Calculate how many questions we can fit
            max_rows = int(available_height / group_height)
            max_groups = max_rows * groups_per_row
            max_questions = max_groups * questions_per_group

            print(f"Warning: Not enough space for {num_questions} questions. Limiting to {max_questions} questions.")
            num_questions = max_questions

            # Recalculate total groups
            total_groups = (num_questions + questions_per_group - 1) // questions_per_group

        if not self.debug:
            # section title, not drawn when debug to avoid overlap
            c.setFont("Helvetica-Bold", 12)
            c.drawString(self.margin, y_start, "ANSWERS")

        y_start -= 1 * cm

        # Draw answer layout
        choices = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # Answer section debug box
        section_height = total_height_needed
        self._draw_debug_box(c, self.margin, y_start - section_height + 1 * cm,
                             available_width, section_height, "(Debug) Answer Section")

        # Current Y position for drawing
        current_y = y_start

        # Draw the groups
        for group_idx in range(total_groups):
            # Calculate group position (row and column)
            row = group_idx // groups_per_row
            col = group_idx % groups_per_row

            # Calculate group boundaries
            group_x = self.margin + (col * (available_width / groups_per_row))
            group_y = current_y - (row * group_height)

            # Calculate first and last question in this group
            first_q = (group_idx * questions_per_group) + 1
            last_q = min((group_idx + 1) * questions_per_group, num_questions)

            if not self.debug:
                # Group header, not drawn when debug to avoid overlap
                c.setFont("Helvetica-Bold", 10)
                c.drawString(group_x, group_y, f"Questions {first_q} - {last_q}")

            # Debug - draw group box
            group_width = available_width / groups_per_row
            actual_group_height = ((last_q - first_q + 1) * question_height) + 0.5 * cm  # Height based on actual questions
            self._draw_debug_box(c,
                                 group_x, group_y - actual_group_height + 0.5 * cm,
                                 group_width, actual_group_height,
                                 f"Group {group_idx + 1}")

            # Draw questions in this group
            question_y = group_y - 0.5 * cm  # Start below group header

            for q_num in range(first_q, last_q + 1):
                # Draw question number
                q_x = group_x + 0.2 * cm  # Small indent
                c.setFont("Helvetica", 10)
                c.drawString(q_x, question_y - 2 * mm, f"{q_num}.")

                # Draw bubbles for each choice horizontally
                for choice_idx in range(choices_per_question):
                    # Position for this choice
                    choice_x = q_x + question_number_width + (choice_idx * choice_width)

                    # Draw the bubble
                    c.circle(choice_x, question_y, self.bubble_radius, stroke=1, fill=0)

                    # Draw the choice letter above the bubble
                    c.drawString(choice_x - self.bubble_radius / 2,
                                 question_y + self.bubble_radius * 1.5,
                                 choices[choice_idx])

                # Move to next question
                question_y -= question_height

        return y_start - section_height + 0.5 * cm

    def _draw_alignment_markers(self, c):
        """Draw alignment markers in the corners for easier scanning."""
        marker_size = 0.5 * cm
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
        """"Draw a straight horizontal line"""
        c.line(self.margin, y_position, self.page_width - self.margin, y_position)


# Example usage
if __name__ == "__main__":
    generator = AnswerSheetGenerator(debug=False)

    # Generate a sample answer sheet with 30 questions, 4 choices each
    filepath, sheet_id, metadata = generator.generate_answer_sheet(num_questions=30, choices_per_question=4)

    print(f"Generated answer sheet: {filepath}")
    print(f"Sheet ID: {sheet_id}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")
