import os
from typing import Optional, Dict, Any, Tuple

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DB_bridging.settings')
import django

django.setup()

from DB_bridging.services import StaticMetricsService, AnswerKeysService, DynamicMetricsService
from DB_bridging.models import StaticMetrics, AnswerKeys, DynamicMetrics


class DatabaseBridge:
    """
    Main API for database operations
    """

    @staticmethod
    def initialize() -> Dict[str, Any]:
        """
        Initialize database and tables
        Call once at application startup
        """
        from django.db import connection

        try:
            # Test connection
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")

            # Create tables
            from django.core.management.commands.migrate import Command
            from django.db import connection
            from django.core.management.color import no_style

            with connection.schema_editor() as schema_editor:
                for model in [StaticMetrics, AnswerKeys, DynamicMetrics]:
                    if not connection.introspection.table_names().__contains__(model._meta.db_table):
                        schema_editor.create_model(model)

            return {
                'status': 'initialized',
                'message': 'Database ready for OMR operations'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    # ========== GENERATION PHASE APIs ==========
    @staticmethod
    def create_static_metrics(static_metrics: StaticMetrics) -> Tuple[StaticMetrics, bool]:
        """
        Create or update static metrics from model instance
        Returns: (instance, created)
        """
        return StaticMetricsService.upsert(static_metrics)

    @staticmethod
    def create_answer_keys(answer_keys: AnswerKeys) -> AnswerKeys:
        """
        Create answer keys from model instance
        Returns: saved instance with auto-generated ID
        """
        return AnswerKeysService.create(answer_keys)

    @staticmethod
    def create_dynamic_metrics(dynamic_metrics: DynamicMetrics) -> DynamicMetrics:
        """
        Create dynamic metrics from model instance
        Fails if instance_id already exists
        """
        return DynamicMetricsService.create(dynamic_metrics)

    @staticmethod
    def create_complete_sheet(
        static_metrics: StaticMetrics,
        answer_keys: AnswerKeys,
        dynamic_metrics: DynamicMetrics
    ) -> Dict[str, Any]:
        """
        Create complete answer sheet (all 3 models) in single transaction
        For generation phase - creates everything needed for answer sheet
        """
        from django.db import transaction

        with transaction.atomic():
            # Create/update static metrics
            static_result = DatabaseBridge.create_static_metrics(static_metrics)

            # Create answer keys
            answer_result = DatabaseBridge.create_answer_keys(answer_keys)

            # Update dynamic metrics with correct foreign keys
            dynamic_metrics.static_template = static_result
            dynamic_metrics.answer_sheet = answer_result

            # Create dynamic metrics
            dynamic_result = DatabaseBridge.create_dynamic_metrics(dynamic_metrics)

            return {
                'static_metrics': static_result,
                'answer_keys': answer_result,
                'dynamic_metrics': dynamic_result,
                'instance_id': dynamic_result.instance_id
            }

    # ========== PROCESSING PHASE APIs ==========

    @staticmethod
    def get_complete_data(instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Get all data for processing phase using dynamic metrics instance_id
        Returns single object with all related data as Python objects
        """
        dynamic_metrics = DynamicMetricsService.get_with_relationships(instance_id)

        if not dynamic_metrics:
            return None

        return {
            'instance_id': dynamic_metrics.instance_id,
            'static_metrics': dynamic_metrics.static_template,
            'answer_keys': dynamic_metrics.answer_sheet,
            'dynamic_metrics': dynamic_metrics,
            'answers': dynamic_metrics.answer_sheet.get_answers(),
        }

    @staticmethod
    def lookup_static_metrics(template_id: str) -> Optional[StaticMetrics]:
        """Lookup static metrics by template_id"""
        return StaticMetricsService.get_by_id(template_id)
