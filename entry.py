import json
from document_generation.layout_one import AnswerSheetGenerator


def generate_document(num_questions: int = 60,
                      questions_per_group: int = 5,
                      choices_per_question: int = 4):
    generator = AnswerSheetGenerator(fill_in=True)
    return generator.generate_answer_sheet(num_questions=num_questions,
                                           choices_per_question=choices_per_question,
                                           questions_per_group=questions_per_group,
                                           filename="deploy.pdf")


if __name__ == "__main__":
    filepath, sheet_id, metadata = generate_document()

    print(f"Generated answer sheet: {filepath}")
    print(f"Sheet ID: {sheet_id}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")
