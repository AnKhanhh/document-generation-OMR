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
