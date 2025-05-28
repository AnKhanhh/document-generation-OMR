import json
from document_generation.layout_one import AnswerSheetGenerator

import os
import fitz  # PyMuPDF


def convert_pdfs_to_images(input_dir, output_dir, format="png", zoom=3):
    """
    Convert PDFs in input_dir to images in output_dir
    zoom: higher = better quality
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all PDF files
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    # Process each PDF
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        base_name = os.path.splitext(pdf_file)[0]

        print(f"Processing: {pdf_file}")

        # Open the PDF
        doc = fitz.open(pdf_path)

        # Process each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Set high resolution rendering
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Output filename
            if len(doc) == 1:
                # For single-page PDFs, don't add page number
                image_path = os.path.join(output_dir, f"{base_name}.{format}")
            else:
                # For multi-page PDFs, add page number
                image_path = os.path.join(output_dir, f"{base_name}_page{page_num + 1}.{format}")

            # Save with Pillow for more options
            pix.pil_save(image_path, optimize=True)

        print(f"Converted {pdf_file} ({len(doc)} pages)")


def generate_document(num_questions: int = 60,
                      questions_per_group: int = 5,
                      choices_per_question: int = 4):
    generator = AnswerSheetGenerator(fill_in=True)
    return generator.generate_answer_sheet(num_questions=num_questions,
                                           choices_per_question=choices_per_question,
                                           questions_per_group=questions_per_group,
                                           filename="deploy.pdf")


if __name__ == "__main__":
    filepath, sheet_id = generate_document()

    print(f"Generated answer sheet: {filepath}")
    print(f"Sheet ID: {sheet_id}")
