from django.db import models
import json
from typing import List, Dict, Any, Optional


class StaticMetrics(models.Model):
    """Template configuration for answer sheets. All distances in reportlab point."""
    template_id = models.CharField(max_length=100, primary_key=True)
    description = models.TextField(blank=True)

    # Layout settings
    page_width = models.IntegerField(null=True, blank=True)
    page_height = models.IntegerField(null=True, blank=True)
    margin = models.IntegerField(null=True, blank=True)
    brush_thickness = models.IntegerField(null=True, blank=True)

    # Marker ID (implicitly ArUco)
    top_left = models.IntegerField(null=True, blank=True)
    top_right = models.IntegerField(null=True, blank=True)
    bottom_right = models.IntegerField(null=True, blank=True)
    bottom_left = models.IntegerField(null=True, blank=True)
    marker_size = models.IntegerField(null=True, blank=True)

    qr_size = models.IntegerField(null=True, blank=True)

    # Text field metrics
    txt_field_width = models.IntegerField(null=True, blank=True)
    txt_field_height = models.IntegerField(null=True, blank=True)
    txt_field_y_spacing = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = 'static_metrics'
        app_label = 'DB_bridging'

    def __str__(self):
        return f"StaticMetrics({self.template_id})"


class AnswerKeys(models.Model):
    """Answer keys with JSON storage for question-answer mappings"""
    # Auto-incrementing primary key (Django default)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)

    # JSON field for answer data
    answer_data = models.JSONField(default=list)

    class Meta:
        db_table = 'answer_keys'
        app_label = 'DB_bridging'

    def __str__(self):
        return f"AnswerKeys({self.id}: {self.name})"

    # JSON conversion methods
    def set_answers(self, answers: List[Dict[str, Any]]) -> None:
        """
        Set answer data from Python objects
        Expected format: [{'question': 1, 'answer': ['A','B'], 'score': 2}, ...]
        """
        self.answer_data = answers

    def get_answers(self) -> List[Dict[str, Any]]:
        """Get answer data as Python objects"""
        return self.answer_data or []

    def add_answer(self, question: int, answer: List[str], score: float = 1.0) -> None:
        """Add a single answer entry"""
        if not self.answer_data:
            self.answer_data = []

        self.answer_data.append({
            'question': question,
            'answer': answer,
            'score': score
        })
        self.total_questions = len(self.answer_data)
        self.total_score = sum(item.get('score', 0) for item in self.answer_data)

    def get_answer_for_question(self, question_num: int) -> Optional[Dict[str, Any]]:
        """Get answer data for specific question"""
        for item in self.get_answers():
            if item.get('question') == question_num:
                return item
        return None


class DynamicMetrics(models.Model):
    """Instance-specific metrics linking template to answer sheet"""
    # Explicit primary key
    instance_id = models.CharField(max_length=100, primary_key=True)
    description = models.TextField(blank=True)

    # Foreign keys
    static_template = models.ForeignKey(
        StaticMetrics,
        on_delete=models.CASCADE,
        related_name='dynamic_instances'
    )
    answer_sheet = models.ForeignKey(
        AnswerKeys,
        on_delete=models.CASCADE,
        related_name='dynamic_instances'
    )

    # Instance-specific metrics
    question_height = models.IntegerField(null=True, blank=True)
    choice_width = models.IntegerField(null=True, blank=True)
    group_y_spacing = models.IntegerField(null=True, blank=True)

    # Template specifications
    num_questions = models.IntegerField(default=20)
    questions_per_group = models.IntegerField(default=10)
    choices_per_question = models.IntegerField(default=4)

    class Meta:
        db_table = 'dynamic_metrics'
        app_label = 'DB_bridging'

    def __str__(self):
        return f"DynamicMetrics({self.instance_id})"
