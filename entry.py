import json

from doccument_generation.layout_one import AnswerSheetGenerator

# Example usage
if __name__ == "__main__":
    debug = False
    generator = AnswerSheetGenerator(debug=debug)
    d_generator = AnswerSheetGenerator()
    num_questions = 60
    questions_per_group = 5
    choices_per_question = 4

    # Generate a sample answer sheet with 30 questions, 4 choices each
    filepath, sheet_id, metadata = \
        generator.generate_answer_sheet(num_questions=num_questions, choices_per_question=choices_per_question, questions_per_group=questions_per_group,
                                        filename="deploy.pdf")

    print(f"Generated answer sheet: {filepath}")
    print(f"Sheet ID: {sheet_id}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")
