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
    def __init__(self, output_dir="./pdf/"):
        self.page_width, self.page_height = A4
        self.margin = 1 * cm
        self.bubble_radius = 3 * mm
        self.bubble_spacing = 10 * mm
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_answer_sheet(self, num_questions: int = 30,
                              choices_per_question: int = 4,
                              sheet_id: int | None = None,
                              filename: str | None = None):
        """
        Generate an answer sheet with the specified number of questions and choices.

        Args:
            num_questions: Number of questions in the test
            choices_per_question: Number of choices per question
            sheet_id: unique ID for answer sheet, generated if not provided
            filename: filename for the PDF, generated if not provided

        Returns:
            tuple: (filename, sheet_id, metadata)
        """
        # Generate parameters if not provided
        if sheet_id is None:
            sheet_id = str(uuid.uuid4())
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"answer_sheet_{timestamp}.pdf"

        filepath = os.path.join(self.output_dir, filename)

        c = canvas.Canvas(filepath, pagesize=A4)

        # Metadata for the answer sheet
        metadata = {
            "sheet_id": sheet_id,
            "num_questions": num_questions,
            "choices_per_question": choices_per_question,
            "created_at": datetime.now().isoformat(),
            "filename": filename
        }

        # Draw the header and title
        self._draw_header(c)

        # Draw form fields (student ID, class, location)
        form_region_y = self._draw_form_fields(c)

        # Draw answer section
        answer_region_y = self._draw_answer_section(c, num_questions, choices_per_question, form_region_y)

        # Draw barcode/QR code with sheet ID
        self._draw_barcode(c, sheet_id, answer_region_y)

        # Draw alignment markers
        self._draw_alignment_markers(c)

        # Save the canvas
        c.save()

        return filepath, sheet_id, metadata

    def _draw_header(self, c: canvas, title: str = "ANSWER SHEET") -> float:
        """Draw the header of the answer sheet."""
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(self.page_width / 2, self.page_height - 2 * cm, title)

        c.setFont("Helvetica", 10)
        c.drawCentredString(self.page_width / 2, self.page_height - 2.5 * cm,
                            "Fill the bubbles completely using a dark pen or pencil")

        # Draw horizontal line
        c.setStrokeColor(colors.black)
        c.line(self.margin, self.page_height - 3 * cm,
               self.page_width - self.margin, self.page_height - 3 * cm)

        return self.page_height - 3.5 * cm

    def _draw_form_fields(self, c: canvas) -> float:
        """Draw the form fields section (student ID, class, location)."""
        y_start = self.page_height - 4 * cm

        # Student ID Field
        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.margin, y_start, "Student ID:")
        c.rect(self.margin + 3 * cm, y_start - 0.5 * cm, 10 * cm, 0.8 * cm)

        # Class Field
        y_start -= 2 * cm
        c.drawString(self.margin, y_start, "Class:")
        c.rect(self.margin + 3 * cm, y_start - 0.5 * cm, 10 * cm, 0.8 * cm)

        # Location Field
        y_start -= 2 * cm
        c.drawString(self.margin, y_start, "Location:")
        c.rect(self.margin + 3 * cm, y_start - 0.5 * cm, 10 * cm, 0.8 * cm)

        # Vietnamese labels
        c.setFont("Helvetica", 8)
        c.drawString(self.margin, y_start + 2.3 * cm, "(Mã số sinh viên)")
        c.drawString(self.margin, y_start + 0.3 * cm, "(Lớp)")
        c.drawString(self.margin, y_start - 1.7 * cm, "(Địa điểm)")

        # Draw horizontal line
        y_start -= 1.5 * cm
        c.setStrokeColor(colors.black)
        c.line(self.margin, y_start,
               self.page_width - self.margin, y_start)

        return y_start - 0.5 * cm

    def _draw_answer_section(self, c: canvas, num_questions: int, choices_per_question: int, y_start: float) -> float:
        """Draw the answer bubbles section."""
        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.margin, y_start, "ANSWERS")

        y_start -= 1 * cm

        # Calculate layout parameters
        questions_per_row = min(5, num_questions)
        num_rows = (num_questions + questions_per_row - 1) // questions_per_row

        # Split into multiple columns if there are many questions
        columns = 1
        if num_questions > 25:
            columns = 2
            questions_per_column = (num_questions + columns - 1) // columns
            questions_per_row = min(5, questions_per_column)
            num_rows = (questions_per_column + questions_per_row - 1) // questions_per_row

        column_width = (self.page_width - 2 * self.margin) / columns

        # Draw column headers
        for col in range(columns):
            col_x = self.margin + col * column_width
            c.setFont("Helvetica-Bold", 10)
            c.drawString(col_x, y_start, f"Questions {col * questions_per_column + 1} - {min((col + 1) * questions_per_column, num_questions)}")

        y_start -= 0.7 * cm

        # Draw the answer bubbles
        choices = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        row_height = choices_per_question * self.bubble_spacing + 0.5 * cm

        for col in range(columns):
            col_start = col * questions_per_column
            col_end = min((col + 1) * questions_per_column, num_questions)
            col_x = self.margin + col * column_width

            for row in range(num_rows):
                for q_idx in range(questions_per_row):
                    question_num = col_start + row * questions_per_row + q_idx + 1

                    if question_num > col_end:
                        continue

                    # Question number
                    q_x = col_x + q_idx * (column_width / questions_per_row)
                    q_y = y_start - row * row_height
                    c.setFont("Helvetica", 10)
                    c.drawString(q_x, q_y, f"{question_num}.")

                    # Draw bubbles for each choice
                    for choice_idx in range(choices_per_question):
                        choice_y = q_y - (choice_idx + 1) * self.bubble_spacing
                        c.circle(q_x + 1 * cm, choice_y, self.bubble_radius, stroke=1, fill=0)
                        c.drawString(q_x + 1 * cm + self.bubble_radius * 1.5,
                                     choice_y - self.bubble_radius / 2,
                                     choices[choice_idx])

        # Calculate the final y position
        final_y = y_start - num_rows * row_height - 1 * cm

        # Draw horizontal line at the bottom of answer section
        c.setStrokeColor(colors.black)
        c.line(self.margin, final_y, self.page_width - self.margin, final_y)

        return final_y - 0.5 * cm

    def _draw_barcode(self, c, sheet_id, y_start):
        """Draw QR code with sheet ID."""
        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.margin, y_start, "SHEET ID")

        # Create QR code
        qr_code = qr.QrCodeWidget(sheet_id)
        bounds = qr_code.getBounds()
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        # Set QR code size
        qr_size = 3 * cm

        # Create drawing for QR code
        d = Drawing(qr_size, qr_size, transform=[qr_size / width, 0, 0, qr_size / height, 0, 0])
        d.add(qr_code)

        # Position and render QR code
        renderPDF.draw(d, c, self.margin, y_start - qr_size - 0.5 * cm)

        # Add sheet ID as text for backup
        c.setFont("Helvetica", 8)
        c.drawString(self.margin, y_start - qr_size - 1 * cm, f"ID: {sheet_id}")

        return y_start - qr_size - 1.5 * cm

    def _draw_alignment_markers(self, c):
        """Draw alignment markers in the corners for easier scanning."""
        marker_size = 0.5 * cm

        # Top-left marker
        c.rect(self.margin, self.page_height - self.margin - marker_size, marker_size, marker_size, fill=1)

        # Top-right marker
        c.rect(self.page_width - self.margin - marker_size, self.page_height - self.margin - marker_size,
               marker_size, marker_size, fill=1)

        # Bottom-left marker
        c.rect(self.margin, self.margin, marker_size, marker_size, fill=1)

        # Bottom-right marker
        c.rect(self.page_width - self.margin - marker_size, self.margin, marker_size, marker_size, fill=1)


# Example usage
if __name__ == "__main__":
    generator = AnswerSheetGenerator()

    # Generate a sample answer sheet with 30 questions, 4 choices each
    filepath, sheet_id, metadata = generator.generate_answer_sheet(num_questions=28, choices_per_question=3)

    print(f"Generated answer sheet: {filepath}")
    print(f"Sheet ID: {sheet_id}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")
