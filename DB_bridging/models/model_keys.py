from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, String, Integer, DateTime, JSON, Text
from sqlalchemy import select, update, delete
from sqlalchemy.orm import relationship

from DB_bridging.database import Base, get_session


@dataclass
class Question:
    """Single question with correct answers and points"""
    correct_answers: List[str]  # e.g. ['A', 'C']
    points: float = 1.0

    def __post_init__(self):
        """Validate answer choices, empty key allowed"""
        if not self.correct_answers:
            raise ValueError("Question must have answer key")

        for answer in self.correct_answers:
            if not (answer.isalpha() and len(answer) == 1):
                raise ValueError(f"Invalid answer choice: {answer}. Must be A-Z")

        self.correct_answers = [s.upper() for s in self.correct_answers]


@dataclass
class AnswerKey:
    """Answer key containing multiple choice questions with validation"""
    questions: Dict[int, Question] = field(default_factory=dict)

    def add_question(self, question_num: int, correct_answers: List[str], points: float = 1.0) -> None:
        """Add a question to the answer key"""
        self.questions[question_num] = Question(correct_answers, points)

    def get_question(self, question_num: int) -> Optional[Question]:
        """Get question by number"""
        return self.questions.get(question_num)

    def remove_question(self, question_num: int) -> bool:
        """Remove question by number"""
        return self.questions.pop(question_num, None) is not None

    def total_points(self) -> float:
        """Calculate total possible points"""
        return sum(q.points for q in self.questions.values())

    def question_count(self) -> int:
        """Get total number of questions"""
        return len(self.questions)

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable format"""
        return {
            str(q_num): {
                "correct_answers": q.correct_answers,
                "points": q.points
            }
            for q_num, q in self.questions.items()
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'AnswerKey':
        """Create AnswerKey from JSON data"""
        questions = {}
        for q_num_str, q_data in data.items():
            q_num = int(q_num_str)
            questions[q_num] = Question(
                correct_answers=q_data["correct_answers"],
                points=q_data.get("points", 1.0)
            )
        return cls(questions=questions)


class AnswerKeys(Base):
    """
    Answer keys unique to each answer sheet
    Primary key: auto-generated integer
    Stores answer data in JSON format
    """
    __tablename__ = 'answer_keys'

    # Auto-generated primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # JSON data for answer key storage
    keys_data = Column(JSON, nullable=False)

    # Relationships
    dynamic_metrics = relationship("DynamicMetrics", back_populates="answer_sheet")

    def __repr__(self):
        return f"<AnswerKeys(id={self.id})>"

    def get_answer_key(self) -> AnswerKey:
        """Get structured answer key from JSON"""
        return AnswerKey.from_json(self.keys_data)

    def set_answer_key(self, answer_key: AnswerKey) -> None:
        """Set answer key from structured object"""
        self.keys_data = answer_key.to_json()


class AnswerKeysRepository:
    """Repository for answer keys with structured data handling"""

    @staticmethod
    def create(answer_key: AnswerKey) -> AnswerKeys:
        """Create new answer keys from AnswerKey object"""
        with get_session() as session:
            answer_keys = AnswerKeys(keys_data=answer_key.to_json())
            session.add(answer_keys)
            session.flush()
            return answer_keys

    @staticmethod
    def get_by_id(answer_id: int) -> Optional[AnswerKeys]:
        """Get answer keys by ID"""
        with get_session() as session:
            return session.get(AnswerKeys, answer_id)

    @staticmethod
    def update_keys(answer_id: int, answer_key: AnswerKey) -> bool:
        """Update answer keys with new AnswerKey object"""
        with get_session() as session:
            result = session.execute(
                update(AnswerKeys)
                .where(AnswerKeys.id == answer_id)
                .values(keys_data=answer_key.to_json())
            )
            return result.rowcount > 0

    @staticmethod
    def get_all() -> List[AnswerKeys]:
        """Get all answer keys"""
        with get_session() as session:
            return list(session.scalars(select(AnswerKeys)))

    @staticmethod
    def delete(answer_id: int) -> bool:
        """Delete answer keys"""
        with get_session() as session:
            result = session.execute(
                delete(AnswerKeys).where(AnswerKeys.id == answer_id)
            )
            return result.rowcount > 0

    @staticmethod
    def get_answer_key_data(answer_id: int) -> Optional[AnswerKey]:
        """Get structured AnswerKey object by ID"""
        answer_keys = AnswerKeysRepository.get_by_id(answer_id)
        return answer_keys.get_answer_key() if answer_keys else None
