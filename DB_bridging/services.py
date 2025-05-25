from typing import Optional, Dict, Any, List, Tuple

from django.db import transaction, IntegrityError

from DB_bridging.models import StaticMetrics, AnswerKeys, DynamicMetrics


class StaticMetricsService:
    """Service for StaticMetrics operations"""

    @staticmethod
    def upsert(static_metrics: StaticMetrics) -> StaticMetrics:
        """Upsert StaticMetrics row"""
        try:
            static_metrics.save()
            return static_metrics
        except IntegrityError:
            print(f"StaticMetrics version '{static_metrics.template_id}' already exists")

    @staticmethod
    def get_by_id(template_id: str) -> Optional[StaticMetrics]:
        """Get StaticMetrics by template_id"""
        try:
            return StaticMetrics.objects.get(template_id=template_id)
        except StaticMetrics.DoesNotExist:
            return None

    @staticmethod
    def create_from_dict(data: Dict[str, Any]) -> StaticMetrics:
        """Create StaticMetrics from dictionary"""
        return StaticMetrics(**data)


class AnswerKeysService:
    """Service for AnswerKeys operations"""

    @staticmethod
    def create(answer_keys: AnswerKeys) -> AnswerKeys:
        """Create new AnswerKeys (auto-generated primary key)"""
        answer_keys.save()
        return answer_keys

    @staticmethod
    def get_by_id(answer_keys_id: int) -> Optional[AnswerKeys]:
        """Get AnswerKeys by ID"""
        try:
            return AnswerKeys.objects.get(id=answer_keys_id)
        except AnswerKeys.DoesNotExist:
            return None

    @staticmethod
    def create_from_answers(name: str, answers: List[Dict[str, Any]], description: str = "") -> AnswerKeys:
        """
        Create AnswerKeys from answer list
        answers format: [{'question': 1, 'answer': ['A','B'], 'score': 2}, ...]
        """
        answer_keys = AnswerKeys(name=name, description=description)
        answer_keys.set_answers(answers)
        answer_keys.save()
        return answer_keys


class DynamicMetricsService:
    """Service for DynamicMetrics operations"""

    @staticmethod
    @transaction.atomic
    def create(dynamic_metrics: DynamicMetrics) -> DynamicMetrics:
        """Upsert DynamicMetrics row"""
        try:
            dynamic_metrics.save()
            return dynamic_metrics
        except IntegrityError:
            raise IntegrityError(f"DynamicMetrics instance_id='{dynamic_metrics.instance_id}' raises integrity error")

    @staticmethod
    def get_by_id(instance_id: str) -> Optional[DynamicMetrics]:
        """Get DynamicMetrics by instance_id"""
        try:
            return DynamicMetrics.objects.get(instance_id=instance_id)
        except DynamicMetrics.DoesNotExist:
            return None

    @staticmethod
    def get_with_relationships(instance_id: str) -> Optional[DynamicMetrics]:
        """
        Get DynamicMetrics with all related data loaded
        Returns single object with static_template and answer_sheet populated
        """
        try:
            return DynamicMetrics.objects.select_related(
                'static_template', 'answer_sheet'
            ).get(instance_id=instance_id)
        except DynamicMetrics.DoesNotExist:
            return None
