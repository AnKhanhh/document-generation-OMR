from dataclasses import dataclass, field
from typing import Tuple, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm


# TODO: magic numbers to avoid potential mem leak between stateful reportlab and stateless streamlit. refactor later
@dataclass
class LayoutConstraints:
    """Holds layout constraints and validation state"""
    # Page dimensions
    _w, _h = A4
    _margin = 1 * cm
    available_height: float = 636.3779527559057 - _margin - 1.2 * cm - 0.2 * cm
    available_width: float = _w - 2 * _margin

    # Component dimensions
    bubble_horizontal_space: float = 1 * cm
    question_number_label_width: float = 0.8 * cm
    question_height: float = 1 * cm
    section_label_height: float = 0.5 * cm
    answer_group_top_margin: float = 0.3 * cm
    lettering_height: float = 0.4 * cm


class AnswerSheetLayoutValidator:
    """Validates answer sheet layout constraints"""

    def __init__(self, constraints: LayoutConstraints):

        self.constraints = constraints

    def validate_choices_per_question(self, choices_per_question: int) -> Tuple[bool, Optional[int]]:
        """
        Validate if choices_per_question fits within width constraints.
        Sets valid_cpq flag in constraints.
        Returns: (is_valid, max_choices_allowed if invalid)
        """
        # Reset validation state for new sequence

        question_number_width = self.constraints.question_number_label_width
        choice_width = self.constraints.bubble_horizontal_space

        # Calculate minimum group width
        min_group_width = question_number_width + (choices_per_question * choice_width)

        if min_group_width <= self.constraints.available_width:
            return True, None
        else:
            # Calculate maximum choices that fit
            max_choices = int((self.constraints.available_width - question_number_width) / choice_width)
            return False, max_choices

    def validate_questions_per_group(self, questions_per_group: int) -> Tuple[bool, Optional[int]]:
        """
        Validate if questions_per_group can fit within height constraints.
        Requires valid_cpq to be True (validate_choices_per_question must be called first).
        Sets valid_qpg flag in constraints.
        Returns: (is_valid, max_questions_per_group if invalid)
        """
        # Reset qpg validation state
        self.constraints.valid_qpg = False

        # Calculate group dimensions
        group_height = self.constraints.question_height * questions_per_group

        # Check if at least one row can fit
        min_height_needed = (group_height +
                             self.constraints.answer_group_top_margin +
                             self.constraints.section_label_height +
                             self.constraints.lettering_height)

        if min_height_needed <= self.constraints.available_height:
            return True, None
        else:
            # Calculate maximum questions per group that fit
            max_questions_per_group = int(
                (self.constraints.available_height - self.constraints.section_label_height -
                 self.constraints.answer_group_top_margin - self.constraints.lettering_height) /
                self.constraints.question_height
            )

            return False, max_questions_per_group

    def validate_questions_num(self, num_questions: int, choices_per_question: int, questions_per_group: int) -> Tuple[bool, Optional[int]]:
        """
        Validate if total number of questions can fit with the validated choices_per_question
        and questions_per_group.
        Requires both valid_cpq and valid_qpg to be True.
        Returns: (is_valid, max_num_questions if invalid)
        """
        # Calculate dimensions
        question_number_width = self.constraints.question_number_label_width
        choice_width = self.constraints.bubble_horizontal_space
        group_width = question_number_width + (choices_per_question * choice_width)
        group_height = self.constraints.answer_group_top_margin + (self.constraints.question_height * questions_per_group)

        # Calculate layout constraints
        max_groups_per_row = int(self.constraints.available_width / group_width)

        # Calculate maximum rows that fit
        max_rows = int(
            (self.constraints.available_height - self.constraints.section_label_height - self.constraints.lettering_height) /
            group_height
        )

        # Calculate maximum total questions
        max_groups = max_rows * max_groups_per_row
        max_total_questions = max_groups * questions_per_group

        if num_questions <= max_total_questions:
            return True, None

        return False, max_total_questions
