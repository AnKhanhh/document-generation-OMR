from typing import List, Dict, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class GradingConfig:
    """Configuration for grading behavior."""
    allow_partial_credit: bool = True
    enable_deduction: bool = False
    deduction_amount: float = 0.0
    deduction_is_relative: bool = False  # If True, deduction_amount is a ratio (0.0-1.0)

    def __post_init__(self):
        if self.deduction_is_relative:
            self.deduction_amount = min(1.0, max(0.0, self.deduction_amount))
        else:
            self.deduction_amount = round(self.deduction_amount)


class QuestionResult(NamedTuple):
    question: int
    student_answer: List[str]
    student_score: float
    state: str
    deduction: float = 0.0


def _calculate_base_score(student_answers: set, correct_answers: set,
                          max_score: int, allow_partial: bool) -> Tuple[float, str]:
    """Calculate base score and state for a question."""
    if not student_answers or len(student_answers) == 0:
        return 0.0, 'empty'

    if len(student_answers) > len(correct_answers):
        return 0.0, 'invalid'

    if student_answers == correct_answers:
        return float(max_score), 'correct'

    if student_answers & correct_answers:
        if allow_partial:
            score = (len(student_answers & correct_answers) / len(correct_answers)) * max_score
            return score, 'partial'
        else:
            return 0.0, 'wrong'

    return 0.0, 'wrong'


def _calculate_deduction(state: str, max_score: int, config: GradingConfig) -> float:
    """Calculate deduction for a question based on its state."""
    if not config.enable_deduction or state not in ['wrong', 'invalid']:
        return 0.0

    if config.deduction_is_relative:
        return config.deduction_amount * max_score
    else:
        return config.deduction_amount


def grade_student_answers(answer_keys: List[Dict[str, int | List[str]]],
                          student_answers: Dict[int, List[str]],
                          config: GradingConfig = None
                          ) -> Tuple[float, List[Dict]]:
    """
    Grade student answers against answer keys with configurable behavior.

    Args:
        answer_keys: List of dicts with "question", "answer" (list), "score" (int)
        student_answers: Dict with question numbers as keys, answer lists as values
        config: GradingConfig object controlling grading behavior

    Returns:
        (score_ratio, grading_details_list)
    """
    if config is None:
        config = GradingConfig()

    total_possible_score = sum(key["score"] for key in answer_keys)
    total_student_score = 0.0
    total_deductions = 0.0
    results = []

    for answer_key in answer_keys:
        question_num = answer_key["question"]
        correct_answers = set(answer_key["answer"])
        max_score = answer_key["score"]
        student_answer = student_answers.get(question_num, [])
        student_answer_set = set(student_answer)

        # Calculate base score
        base_score, state = _calculate_base_score(
            student_answer_set, correct_answers, max_score, config.allow_partial_credit
        )

        # Calculate deduction
        deduction = _calculate_deduction(state, max_score, config)

        # Final score for this question
        final_score = base_score - deduction

        total_student_score += base_score
        total_deductions += deduction

        results.append({
            "question": question_num,
            "student_answer": student_answer,
            "student_score": final_score,
            "state": state,
            "deduction": deduction
        })

    # Apply total deductions and ensure score doesn't go below 0
    final_total_score = max(0.0, total_student_score - total_deductions)
    ratio = final_total_score / total_possible_score if total_possible_score > 0 else 0.0

    return ratio, results


# Example usage:
# if __name__ == "__main__":
#     answer_keys = [
#         {"question": 1, "answer": ["A", "B"], "score": 10},
#         {"question": 2, "answer": ["C"], "score": 5},
#     ]
#
#     student_answers = {
#         1: ["A"],  # Partial credit
#         2: ["D"]  # Wrong answer
#     }
#
#     configs = [
#         GradingConfig(),  # Default behavior
#         GradingConfig(allow_partial_credit=False),  # No partial credit
#         GradingConfig(enable_deduction=True, deduction_amount=1.0),  # Fixed deduction
#         GradingConfig(enable_deduction=True, deduction_amount=0.1, deduction_is_relative=True),  # Relative deduction
#     ]
#
#     for i, config in enumerate(configs):
#         ratio, details = grade_student_answers(answer_keys, student_answers, config)
#         print(f"Config {i + 1}: Score ratio = {ratio:.2f}")
#         for detail in details:
#             print(f"  Q{detail['question']}: {detail['state']}, score={detail['student_score']}, deduction={detail['deduction']}")
#         print()
