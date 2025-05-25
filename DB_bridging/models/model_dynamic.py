from datetime import datetime
from typing import Optional, List

from sqlalchemy import Column, String, Integer, ForeignKey, DateTime
from sqlalchemy import select, update, delete
from sqlalchemy.orm import relationship

from DB_bridging.database import Base, get_session
from DB_bridging.models.model_keys import AnswerKeys, AnswerKeysRepository
from DB_bridging.models.model_static import StaticMetrics, StaticMetricsRepository


class DynamicMetrics(Base):
    """
    Dynamic metrics unique to each answer sheet instance
    Primary key: explicitly defined string (instance_id)
    Links to both static_metrics (template) and answer_keys (sheet)
    """
    __tablename__ = 'dynamic_metrics'

    # Explicit string primary key (instance identifier)
    instance_id = Column(String(100), primary_key=True)

    # Foreign keys
    static_metrics_id = Column(String(100), ForeignKey('static_metrics.template_id'), nullable=False)
    answer_keys_id = Column(Integer, ForeignKey('answer_keys.id'), nullable=False)

    # Answer sheet properties
    num_questions = Column(Integer, nullable=True)
    question_per_group = Column(Integer, nullable=True)
    answer_per_question = Column(Integer, nullable=True)

    # Detection metrics
    detected_questions = Column(Integer, nullable=True)
    detected_answers = Column(Integer, nullable=True)
    confidence_score = Column(Integer, nullable=True)  # 0-100

    # Position adjustments from template
    x_offset = Column(Integer, nullable=True)
    y_offset = Column(Integer, nullable=True)
    scale_factor_x = Column(Integer, nullable=True)  # Store as integer (multiply by 1000)
    scale_factor_y = Column(Integer, nullable=True)  # Store as integer (multiply by 1000)

    # Processing metrics
    processing_time_ms = Column(Integer, nullable=True)
    error_count = Column(Integer, nullable=True)

    # Additional scalable metrics
    custom_dynamic_1 = Column(Integer, nullable=True)
    custom_dynamic_2 = Column(Integer, nullable=True)
    custom_dynamic_3 = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    static_template = relationship("StaticMetrics", back_populates="dynamic_metrics")
    answer_sheet = relationship("AnswerKeys", back_populates="dynamic_metrics")

    def __repr__(self):
        return f"<DynamicMetrics(instance_id='{self.instance_id}', status='{self.processing_status}')>"


class DynamicMetricsRepository:
    """Repository for dynamic metrics with upsert capabilities"""

    @staticmethod
    def create_or_update(
        instance_id: str,
        static_metrics_id: str,
        answer_keys_id: int,
        instance_name: str,
        **metrics
    ) -> DynamicMetrics:
        """Create new or update existing dynamic metrics (upsert)"""
        with get_session() as session:
            # Try to get existing record
            existing = session.get(DynamicMetrics, instance_id)

            if existing:
                # Update existing record
                for key, value in metrics.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.static_metrics_id = static_metrics_id
                existing.answer_keys_id = answer_keys_id
                existing.instance_name = instance_name
                existing.updated_at = datetime.utcnow()
                session.flush()
                return existing
            else:
                # Create new record
                metric = DynamicMetrics(
                    instance_id=instance_id,
                    static_metrics_id=static_metrics_id,
                    answer_keys_id=answer_keys_id,
                    instance_name=instance_name,
                    **metrics
                )
                session.add(metric)
                session.flush()
                return metric

    @staticmethod
    def get_by_id(instance_id: str) -> Optional[DynamicMetrics]:
        """Get dynamic metrics by instance ID"""
        with get_session() as session:
            return session.get(DynamicMetrics, instance_id)

    @staticmethod
    def get_by_answer_sheet(answer_keys_id: int) -> List[DynamicMetrics]:
        """Get all dynamic metrics for an answer sheet"""
        with get_session() as session:
            stmt = select(DynamicMetrics).where(
                DynamicMetrics.answer_keys_id == answer_keys_id
            ).order_by(DynamicMetrics.created_at.desc())
            return list(session.scalars(stmt))

    @staticmethod
    def get_with_relationships(instance_id: str) -> Optional[DynamicMetrics]:
        """Get dynamic metrics with loaded relationships"""
        with get_session() as session:
            stmt = select(DynamicMetrics).where(
                DynamicMetrics.instance_id == instance_id
            ).join(StaticMetrics).join(AnswerKeys)
            return session.scalar(stmt)

    @staticmethod
    def update_status(instance_id: str, status: str) -> bool:
        """Update processing status"""
        with get_session() as session:
            stmt = update(DynamicMetrics).where(
                DynamicMetrics.instance_id == instance_id
            ).values(processing_status=status, updated_at=datetime.utcnow())
            result = session.execute(stmt)
            return result.rowcount > 0

    @staticmethod
    def delete(instance_id: str) -> bool:
        """Delete dynamic metrics"""
        with get_session() as session:
            stmt = delete(DynamicMetrics).where(DynamicMetrics.instance_id == instance_id)
            result = session.execute(stmt)
            return result.rowcount > 0
