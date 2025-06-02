from DB_bridging.database_bridge import DatabaseBridge
import cv2

import document_extraction.extraction_entry as extraction
from document_generation.layout_one import AnswerSheetGenerator
import os
import fitz  # PyMuPDF


def convert_pdfs_to_images(input_dir, output_dir, ext="png", zoom=3):
    """Convert PDFs images. higher zoom = better quality"""
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
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Output filename
            if len(doc) == 1:
                image_path = os.path.join(output_dir, f"{base_name}.{ext}")
            else:
                image_path = os.path.join(output_dir, f"{base_name}_page{page_num + 1}.{ext}")

            pix.pil_save(image_path, optimize=True)

        print(f"Converted {pdf_file} ({len(doc)} pages)")


def generate_document(num_questions: int = 60,
                      questions_per_group: int = 5,
                      choices_per_question: int = 4,
                      keys=None):
    from DB_bridging.models import AnswerKeys
    from DB_bridging.id_gen import IDGenerator

    id_generator = IDGenerator()
    sheet_id = id_generator.generate()

    generator = AnswerSheetGenerator()
    path, _ = generator.generate_answer_sheet(num_questions=num_questions,
                                              choices_per_question=choices_per_question,
                                              questions_per_group=questions_per_group,
                                              sheet_id=sheet_id,
                                              filename="deploy.pdf")
    if keys is None:
        keys = []
    answer_keys = AnswerKeys()
    answer_keys.set_answers(keys)

    db_metrics_log = DatabaseBridge.create_complete_sheet(
        generator.static_metrics,
        answer_keys,
        generator.dynamic_metrics
    )
    print(f"Saved answer sheet metrics to DB,"
          f" sheet ID = {db_metrics_log['dynamic_metrics']},"
          f" layout = {db_metrics_log['static_metrics']},"
          f" answers ID = {db_metrics_log['answer_keys']}")

    convert_pdfs_to_images("out/pdf", "out/image", ext="png", zoom=3)

    return path, sheet_id


def generate_lab_test(num_questions: int = 47,
                      questions_per_group: int = 4,
                      choices_per_question: int = 5):
    from DB_bridging.id_gen import IDGenerator
    sheet_id = IDGenerator().generate()

    AnswerSheetGenerator(fill_in=True) \
        .generate_answer_sheet(num_questions=num_questions,
                               choices_per_question=choices_per_question,
                               questions_per_group=questions_per_group,
                               sheet_id=sheet_id,
                               filename="filled.pdf")

    template_sheet = AnswerSheetGenerator()
    template_sheet.generate_answer_sheet(num_questions=num_questions,
                                         choices_per_question=choices_per_question,
                                         questions_per_group=questions_per_group,
                                         sheet_id=sheet_id,
                                         filename="pristine.pdf")
    convert_pdfs_to_images("out/pdf", "out/image", ext="png", zoom=3)

    from misc import generate_answer_keys
    from DB_bridging.models import AnswerKeys
    answer_keys = AnswerKeys()
    answer_keys.set_answers(generate_answer_keys(num_questions, questions_per_group))

    db_metrics_log = DatabaseBridge.create_complete_sheet(
        template_sheet.static_metrics,
        answer_keys,
        template_sheet.dynamic_metrics
    )
    print(f"Saved answer sheet metrics to DB,"
          f" sheet ID = {db_metrics_log['dynamic_metrics']},"
          f" layout = {db_metrics_log['static_metrics']},"
          f" answers ID = {db_metrics_log['answer_keys']}")

    from document_extraction.distortion import DocumentDistorter
    template = cv2.imread("out/image/pristine.png", cv2.IMREAD_COLOR)
    distorted = cv2.imread("out/image/filled.png", cv2.IMREAD_COLOR)
    distorter = DocumentDistorter()
    distorted = distorter.apply_perspective_distortion(distorted, severity=0.6)
    distorted = distorter.apply_rotation(distorted, angle=50)
    distorted = distorter.apply_lighting_variation(distorted, contrast_factor=1, max_shadow=0.5)
    distorted = distorter.apply_noise(distorted, 0.1)
    cv2.imwrite("out/image/filled_distorted.png", distorted)


def clean_directory(directory: str) -> None:
    """Delete all files in directory, printing each filename."""
    from pathlib import Path
    dir_path = Path(directory)
    if not dir_path.exists():
        return

    print(f"Cleaning up {directory}:", end="\t")

    count = 0
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            count += 1
            print(f"{file_path.name}", end=" ")
            file_path.unlink()

    print(f"...{count} files deleted")


if __name__ == "__main__":
    init_result = DatabaseBridge.initialize()
    print(init_result)

    os.makedirs("out/image", exist_ok=True)
    os.makedirs("out/vis_detection", exist_ok=True)
    os.makedirs("out/pdf", exist_ok=True)

    clean_directory("out/image")
    clean_directory("out/vis_detection")
    clean_directory("out/pdf")

    generate_lab_test()

    photo = cv2.imread("out/image/filled_distorted.png", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("out/image/pristine.png", cv2.IMREAD_GRAYSCALE)
    init_result = DatabaseBridge.initialize()
    warped, viz = extraction.extract(photo, template, visualize=True)

    cv2.imwrite("out/image/filled_corrected.png",warped)

    for k, v in viz.items():
        cv2.imwrite(f"out/vis_detection/{k}.png", v)
