from dataclasses import dataclass, field
from typing import Tuple, Optional

from reportlab.lib.units import cm


@dataclass
class LayoutConstraints:
    """Holds layout constraints and validation state"""
    # Page dimensions
    available_height: float
    available_width: float

    # Component dimensions
    bubble_horizontal_space: float
    question_number_label_width: float
    question_height: float
    section_label_height: float
    answer_group_top_margin: float
    lettering_height: float

    # Validation state flags (reset on each validation sequence)
    valid_cpq: bool = field(default=False, init=False)  # choices_per_question validated
    valid_qpg: bool = field(default=False, init=False)  # questions_per_group validated

    # Store validated values for subsequent validation steps
    _validated_cpq: int = field(default=None, init=False)
    _validated_qpg: int = field(default=None, init=False)


class AnswerSheetLayoutValidator:
    """Validates answer sheet layout constraints"""

    def __init__(self, constraints: LayoutConstraints):

        self.constraints = constraints

    def reset_validation_state(self):
        """Reset validation flags for a new validation sequence"""
        self.constraints.valid_cpq = False
        self.constraints.valid_qpg = False
        self.constraints._validated_cpq = None
        self.constraints._validated_qpg = None

    def validate_choices_per_question(self, choices_per_question: int) -> Tuple[bool, Optional[int]]:
        """
        Validate if choices_per_question fits within width constraints.
        Sets valid_cpq flag in constraints.
        Returns: (is_valid, max_choices_allowed if invalid)
        """
        # Reset validation state for new sequence
        self.reset_validation_state()

        question_number_width = self.constraints.question_number_label_width
        choice_width = self.constraints.bubble_horizontal_space

        # Calculate minimum group width
        min_group_width = question_number_width + (choices_per_question * choice_width)

        if min_group_width <= self.constraints.available_width:
            # Valid - set flag and store for later use
            self.constraints.valid_cpq = True
            self.constraints._validated_cpq = choices_per_question
            return True, None

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
        # Check pipeline sequence
        if not self.constraints.valid_cpq:
            raise ValueError("Must call validate_choices_per_question first")

        # Reset qpg validation state
        self.constraints.valid_qpg = False

        choices_per_question = self.constraints._validated_cpq

        # Calculate group dimensions
        question_number_width = self.constraints.question_number_label_width
        choice_width = self.constraints.bubble_horizontal_space
        group_width = question_number_width + (choices_per_question * choice_width)
        group_height = self.constraints.question_height * questions_per_group

        # Check if at least one row can fit
        min_height_needed = (
            group_height +
            self.constraints.answer_group_top_margin +
            self.constraints.section_label_height +
            self.constraints.lettering_height
        )

        if min_height_needed <= self.constraints.available_height:
            # Valid - set flag and store for later use
            self.constraints.valid_qpg = True
            self.constraints._validated_qpg = questions_per_group
            return True, None

        # Calculate maximum questions per group that fit
        max_questions_per_group = int(
            (self.constraints.available_height -
             self.constraints.section_label_height -
             self.constraints.answer_group_top_margin -
             self.constraints.lettering_height) /
            self.constraints.question_height
        )

        return False, max_questions_per_group

    def validate_questions_layout(self, num_questions: int) -> Tuple[bool, Optional[int]]:
        """
        Validate if total number of questions can fit with the validated choices_per_question
        and questions_per_group.
        Requires both valid_cpq and valid_qpg to be True.
        Returns: (is_valid, max_num_questions if invalid)
        """
        # Check pipeline sequence
        if not self.constraints.valid_cpq:
            raise ValueError("Must call validate_choices_per_question first")
        if not self.constraints.valid_qpg:
            raise ValueError("Must call validate_questions_per_group first")

        choices_per_question = self.constraints._validated_cpq
        questions_per_group = self.constraints._validated_qpg

        # Calculate dimensions
        question_number_width = self.constraints.question_number_label_width
        choice_width = self.constraints.bubble_horizontal_space
        group_width = question_number_width + (choices_per_question * choice_width)
        group_height = self.constraints.question_height * questions_per_group

        # Calculate layout constraints
        max_groups_per_row = int(self.constraints.available_width / group_width)

        # Calculate maximum rows that fit
        max_rows = int(
            (self.constraints.available_height -
             self.constraints.section_label_height -
             self.constraints.lettering_height) /
            (group_height + self.constraints.answer_group_top_margin)
        )

        # Calculate maximum total questions
        max_groups = max_rows * max_groups_per_row
        max_total_questions = max_groups * questions_per_group

        if num_questions <= max_total_questions:
            return True, None

        return False, max_total_questions


# Example usage for external validation with user prompts
def validate_answer_sheet_parameters(self, y_start: float) -> Tuple[int, int, int]:
    """
    Example function showing how to validate parameters with user prompts.
    Returns validated (num_questions, choices_per_question, questions_per_group)
    """
    # Create constraints
    available_height = y_start - (self.margin + self.marker_size + 0.2 * cm)
    available_width = self.page_width - 2 * self.margin

    constraints = LayoutConstraints(
        available_height=available_height,
        available_width=available_width,
        bubble_horizontal_space=self.bubble_horizontal_space,
        question_number_label_width=self.question_number_label_width,
        question_height=self.question_height,
        section_label_height=self.section_label_height,
        answer_group_top_margin=self.answer_group_top_margin,
        lettering_height=self.lettering_height
    )

    validator = AnswerSheetLayoutValidator(constraints)

    # Validate choices per question
    while True:
        choices_per_question = int(input("Enter choices per question: "))
        valid, max_choices = validator.validate_choices_per_question(choices_per_question)
        if valid:
            break
        print(f"Invalid. Maximum choices allowed: {max_choices}")

    # Validate questions per group
    while True:
        questions_per_group = int(input("Enter questions per group: "))
        valid, max_qpg = validator.validate_questions_per_group(questions_per_group)
        if valid:
            break
        print(f"Invalid. Maximum questions per group allowed: {max_qpg}")

    # Validate total questions
    while True:
        num_questions = int(input("Enter total number of questions: "))
        valid, max_questions = validator.validate_questions_layout(num_questions)
        if valid:
            break
        print(f"Invalid. Maximum questions allowed: {max_questions}")

    return num_questions, choices_per_question, questions_per_group


def validate_b(value):
    """Validate choices parameter"""
    return value < 100


def validate_c(value):
    """Validate groupings parameter"""
    return value < 100


def validate_a(value, b_val, c_val):
    """Validate questions parameter based on b and c values"""
    # Example: questions must be <= b * c
    return value <= (b_val * c_val) * 10
