import cv2

import document_extraction.extraction_entry as extraction
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

    # Process PDFs
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        base_name = os.path.splitext(pdf_file)[0]

        print(f"Processing: {pdf_file}")

        # Process pages
        doc = fitz.open(pdf_path)
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
                image_path = os.path.join(output_dir, f"{base_name}_page{page_num + 1}.{format}")

            # Save with Pillow for more options
            pix.pil_save(image_path, optimize=True)

        print(f"Converted {pdf_file} ({len(doc)} pages)")


def generate_document(num_questions: int = 60,
                      questions_per_group: int = 5,
                      choices_per_question: int = 4):
    from DB_bridging.models import AnswerKeys
    from DB_bridging.database_bridge import DatabaseBridge
    from DB_bridging.id_gen import IDGenerator

    id_generator = IDGenerator()
    sheet_id = id_generator.generate()

    generator = AnswerSheetGenerator(fill_in=True)
    path, id = generator.generate_answer_sheet(num_questions=num_questions,
                                               choices_per_question=choices_per_question,
                                               questions_per_group=questions_per_group,
                                               sheet_id=sheet_id,
                                               filename="deploy.pdf")
    answer_keys = AnswerKeys()
    answer_keys.set_answers([])
    db_metrics_log = DatabaseBridge.create_complete_sheet(
        generator.static_metrics,
        answer_keys,
        generator.dynamic_metrics
    )

    return path, id


if __name__ == "__main__":
    from DB_bridging.database_bridge import DatabaseBridge

    init_result = DatabaseBridge.initialize()
    print(init_result)

    # filepath, id = generate_document()
    # convert_pdfs_to_images("out/pdf", "out/image", format="png", zoom=3)

    photo = cv2.imread("out/image/filled_distorted.png", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("out/image/pristine.png", cv2.IMREAD_GRAYSCALE)
    init_result = DatabaseBridge.initialize()
    viz = extraction.extract(photo, template, visualize=True)

    os.makedirs("out/vis_detection", exist_ok=True)
    for k, v in viz.items():
        cv2.imwrite(f"out/vis_detection/{k}.png", v)
