import json

from doccument_generation.layout_one import AnswerSheetGenerator

# Example usage
if __name__ == "__main__":
    generator = AnswerSheetGenerator()

    # Generate a sample answer sheet with 30 questions, 4 choices each
    filepath, sheet_id, metadata = generator.generate_answer_sheet(num_questions=30, choices_per_question=4)

    print(f"Generated answer sheet: {filepath}")
    print(f"Sheet ID: {sheet_id}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")
