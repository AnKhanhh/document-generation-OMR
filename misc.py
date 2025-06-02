from typing import Dict, List, Tuple

import numpy as np


def generate_answer_keys(num_questions, choices_per_question):
    rng = np.random.default_rng()
    choices = [chr(ord('A') + i) for i in range(choices_per_question)]

    result = []
    for i in range(num_questions):
        # 80% chance single answer, 20% chance multiple
        num_answers = 1 if rng.random() < 0.8 else rng.integers(2, choices_per_question)
        answers = sorted(rng.choice(choices, size=num_answers, replace=False))

        result.append({
            'question': i + 1,
            'answer': [str(s) for s in answers],
            'score': int(rng.integers(1, 5))
        })

    return result


def grade_student_answers(answer_keys: List[Dict[str, int | List[str]]],
                          student_answers: Dict[int, List[str]]
                          ) -> Tuple[float, List[Dict[str, int | List[str] | str]]]:
    """
    Grade student answers against answer keys.
    Args:
        answer_keys: List of dicts with "question", "answer" (list), "score" (int)
        student_answers: Dict with question numbers as keys, answer lists as values
    Returns:
        (score, grading_details_list)
    """
    total_possible_score = sum(key["score"] for key in answer_keys)
    total_student_score = 0
    grading_details = []

    for answer_key in answer_keys:
        question_num = answer_key["question"]
        correct_answers = set(answer_key["answer"])
        max_score = answer_key["score"]

        student_answer = student_answers.get(question_num, [])
        student_answer_set = set(student_answer)

        # Apply grading rules
        if not student_answer:  # Rule 1: empty answer
            score, state = 0, 'empty'
        elif len(student_answer) > len(correct_answers):  # Rule 2: too many answers
            score, state = 0, 'invalid'
        elif student_answer_set == correct_answers:  # Rule 3: perfect match
            score, state = max_score, 'correct'
        elif student_answer_set & correct_answers:  # Rule 4: partial match
            score = (len(student_answer_set & correct_answers) / len(correct_answers)) * max_score
            state = 'partial'
        else:  # Rule 5: no correct answers
            score, state = 0, 'wrong'

        total_student_score += score
        grading_details.append({
            "question": question_num,
            "student_answer": student_answer,
            "student_score": score,
            "state": state
        })

    ratio = (total_student_score / total_possible_score) if total_possible_score > 0 else 0
    return ratio, grading_details
