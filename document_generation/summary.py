import cv2
import numpy as np
from io import BytesIO
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors


class SummaryGeneration:
    def __init__(self, answer_sheet_id, answer_keys, student_grading, final_score,
                 answer_sheet_image, bounding_boxes):
        self.answer_sheet_id = answer_sheet_id
        self.answer_keys = answer_keys
        self.student_grading = student_grading
        self.final_score = final_score
        self.answer_sheet_image = answer_sheet_image
        self.bounding_boxes = bounding_boxes
        self.styles = getSampleStyleSheet()

    def _crop_image_from_bbox(self, bbox):
        """Crop image using bounding box coordinates (top-left clockwise)"""
        # Extract coordinates
        tl, tr, br, bl = bbox

        # Get bounding rectangle
        x_min = min(tl[0], tr[0], br[0], bl[0])
        y_min = min(tl[1], tr[1], br[1], bl[1])
        x_max = max(tl[0], tr[0], br[0], bl[0])
        y_max = max(tl[1], tr[1], br[1], bl[1])

        # Crop the image
        cropped = self.answer_sheet_image[y_min:y_max, x_min:x_max]
        return self._cv2_to_reportlab_image(cropped)

    @staticmethod
    def _cv2_to_reportlab_image(cv_image, max_width=2 * inch):
        """Convert OpenCV image to ReportLab Image object"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        pil_image = PILImage.fromarray(rgb_image)

        # Save to Buffer
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        # Calculate dimensions maintaining aspect ratio
        aspect_ratio = pil_image.height / pil_image.width
        width = max_width
        height = width * aspect_ratio

        return Image(img_buffer, width=width, height=height)

    def _create_header(self):
        """Create document header with answer sheet ID and final score"""
        elements = []

        # Title
        title_style = ParagraphStyle('CustomTitle', parent=self.styles['Heading1'],
                                     fontSize=16, spaceAfter=12)
        elements.append(Paragraph("Grading Summary", title_style))

        # Answer sheet ID and final score
        info_style = ParagraphStyle('Info', parent=self.styles['Normal'],
                                    fontSize=12, spaceAfter=6)
        elements.append(Paragraph(f"<b>Answer Sheet ID:</b> {self.answer_sheet_id}", info_style))
        elements.append(Paragraph(f"<b>Final Score:</b> {self.final_score * 100:.1f}%", info_style))
        elements.append(Spacer(1, 12))

        return elements

    def _create_cropped_images_section(self):
        """Create section with cropped images from bounding boxes"""
        elements = []

        # Section header
        header_style = ParagraphStyle('SectionHeader', parent=self.styles['Heading2'],
                                      fontSize=14, spaceAfter=8)
        elements.append(Paragraph("Answer Sheet Sections", header_style))

        # Create images
        labels = ["Student ID", "Class", "Location"]  # Placeholder labels

        for i, bbox in enumerate(self.bounding_boxes):
            img = self._crop_image_from_bbox(bbox)
            elements.append(Paragraph(f"<b>{labels[i]}:</b>", self.styles['Normal']))
            elements.append(img)
            elements.append(Spacer(1, 8))

        return elements

    def _create_grading_table(self):
        """Create table showing question-by-question grading results"""
        elements = []

        # Section header
        header_style = ParagraphStyle('SectionHeader', parent=self.styles['Heading2'],
                                      fontSize=14, spaceAfter=8)
        elements.append(Paragraph("Question Results", header_style))

        # Prepare table data
        data = [['Q#', 'Correct Answer', 'Student Answer', 'Max Score', 'Score', 'Status']]

        for i, (key, student) in enumerate(zip(self.answer_keys, self.student_grading)):
            correct_ans = ', '.join(key['answer'])
            student_ans = ', '.join(student['student_answer'])

            data.append([
                str(key['question']),
                correct_ans,
                student_ans,
                str(key['score']),
                str(student['student_score']),
                student['state'].title()
            ])

        # Create table
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(table)
        return elements

    def generate_pdf(self, filename):
        """Generate the complete PDF summary"""
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []

        # Build document
        story.extend(self._create_header())
        story.extend(self._create_cropped_images_section())
        story.append(Spacer(1, 20))
        story.extend(self._create_grading_table())

        doc.build(story)
        print(f"PDF generated: {filename}")

# Example:
# summary = SummaryGeneration(
#     answer_sheet_id="12345",
#     answer_keys=[{"question": 1, "answer": ["C"], "score": 4}],
#     student_grading=[{"question": 1, "student_answer": ["A"], "student_score": 0, "state": "wrong"}],
#     final_score=0.75,
#     answer_sheet_image=cv_image,
#     bounding_boxes=[bbox1, bbox2, bbox3]
# )
# summary.generate_pdf("student_summary.pdf")
