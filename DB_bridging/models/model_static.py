from datetime import datetime
from typing import Optional, List

from sqlalchemy import Column, String, Integer, DateTime, Text
from sqlalchemy import select, delete
from sqlalchemy.orm import relationship

from DB_bridging.database import Base, get_session


class StaticMetrics(Base):
    """
    Static metrics tied to template versions - persistent across answer sheets
    Primary key: explicitly defined string (template_id)
    Stores reportlab document points for layout calculations
    """
    __tablename__ = 'static_metrics'

    # Explicit string primary key
    template_id = Column(String(30), primary_key=True)

    # Alignment marker IDs, implicitly DICT_6X6_250 ArUco
    top_left = Column(Integer, nullable=True)
    top_right = Column(Integer, nullable=True)
    bottom_right = Column(Integer, nullable=True)
    bottom_left = Column(Integer, nullable=True)

    # ReportLab generation metrics, in document point
    # Global metrics
    page_width = Column(Integer, nullable=True)
    page_height = Column(Integer, nullable=True)
    margin = Column(Integer, nullable=True)
    brush_thickness = Column(Integer, nullable=True)

    # Text section metrics
    txt_field_width = Column(Integer, nullable=True)
    txt_field_height = Column(Integer, nullable=True)
    txt_field_y_spacing = Column(Integer, nullable=True)

    # Markers metrics
    marker_size = Column(Integer, nullable=True)
    qr_size = Column(Integer, nullable=True)

    # Relationships
    dynamic_metrics = relationship("DynamicMetrics", back_populates="static_template")

    def __repr__(self):
        return f"<StaticMetrics(template_id='{self.template_id}')>"


class StaticMetricsRepository:
    """Repository for static metrics with upsert capabilities"""

    @staticmethod
    def create_or_update(metrics_instance: StaticMetrics) -> StaticMetrics:
        """Upsert static metrics using merge for simplicity"""
        with get_session() as session:
            # Modern SQLAlchemy approach: merge handles upsert automatically
            merged = session.merge(metrics_instance)
            session.commit()
            return merged

    @staticmethod
    def get_by_id(template_id: str) -> Optional[StaticMetrics]:
        """Get static metrics by template ID"""
        with get_session() as session:
            return session.get(StaticMetrics, template_id)

    @staticmethod
    def get_all() -> List[StaticMetrics]:
        """Get all static metrics"""
        with get_session() as session:
            return list(session.scalars(select(StaticMetrics).order_by(StaticMetrics.template_id)))

    @staticmethod
    def delete(template_id: str) -> bool:
        """Delete static metrics"""
        with get_session() as session:
            result = session.execute(
                delete(StaticMetrics).where(StaticMetrics.template_id == template_id)
            )
            session.commit()
            return result.rowcount > 0
