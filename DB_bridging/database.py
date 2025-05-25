import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Any

from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Configure logging for SQL queries
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)


class DatabaseConfig:
    """Database configuration settings"""

    def __init__(
        self,
        db_path: str = "./DB/",
        db_name: str = "app.sqlite",
        echo: bool = False,
        check_same_thread: bool = False
    ):
        self.db_path = Path(db_path)
        self.db_name = db_name
        self.echo = echo
        self.check_same_thread = check_same_thread

        # Ensure directory exists and build connection URL
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.database_url = f"sqlite:///{self.db_path / self.db_name}"

    @property
    def is_sqlite(self) -> bool:
        """Check if database is SQLite"""
        return "sqlite" in self.database_url


class Base(DeclarativeBase):
    """Modern declarative base with metadata configuration"""
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",                                          # Index names
            "uq": "uq_%(table_name)s_%(column_0_name)s",                            # Unique constraints
            "ck": "ck_%(table_name)s_%(constraint_name)s",                          # Check constraints
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",    # Foreign keys
            "pk": "pk_%(table_name)s"                                               # Primary keys
        }
    )


class DatabaseManager:
    """Central database management class"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False
        )

    def _create_engine(self):
        """Create database engine with optimal settings"""
        engine_kwargs = {
            "echo": self.config.echo,
            "future": True  # Use SQLAlchemy 2.x behaviors
        }

        if self.config.is_sqlite:
            engine_kwargs.update({
                "connect_args": {
                    "check_same_thread": self.config.check_same_thread,
                    "isolation_level": None,  # Autocommit mode
                },
                "poolclass": StaticPool
            })

        return create_engine(self.config.database_url, **engine_kwargs)

    # === SESSION MANAGEMENT API ===

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Primary session context manager"""
        with self.SessionLocal() as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    @contextmanager
    def get_transaction(self) -> Generator[Session, None, None]:
        """Explicit transaction control"""
        with self.SessionLocal() as session:
            with session.begin():
                yield session

    def get_raw_session(self) -> Session:
        """Raw session, require manual commit/rollback/close"""
        return self.SessionLocal()

    # === DATABASE LIFECYCLE API ===

    def create_all_tables(self) -> None:
        """Create all tables from registered models"""
        Base.metadata.create_all(self.engine)
        print("Missing tables created")

    @property
    def registered_models(self) -> list[str]:
        """List all currently registered model tables"""
        return list(Base.metadata.tables.keys())

    def drop_all_tables(self) -> None:
        """Drop all tables"""
        Base.metadata.drop_all(self.engine)

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        with self.get_session() as session:
            result = session.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"),
                {"table_name": table_name}
            )
            return result.fetchone() is not None

    def get_table_info(self, table_name: str) -> list[Any]:
        """Get table schema information"""
        with self.get_session() as session:
            return session.execute(text(f"PRAGMA table_info({table_name})")).fetchall()

    # === UTILITY API ===

    def execute_raw_sql(self, sql: str, params: dict[str, Any] | None = None) -> list[Any]:
        """Execute raw SQL with parameters"""
        with self.get_session() as session:
            return session.execute(text(sql), params or {}).fetchall()

    @property
    def database_info(self) -> dict[str, Any]:
        """Get database metadata"""
        db_file = self.config.db_path / self.config.db_name
        file_size_mb = round(db_file.stat().st_size / (1024 * 1024), 2) if db_file.exists() else 0

        # Get table list
        with self.get_session() as session:
            tables = [
                row[0] for row in session.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                ).fetchall()
            ]

        info = {
            "database_url": self.config.database_url,
            "database_path": str(db_file),
            "file_size_mb": file_size_mb,
            "tables": tables
        }
        return info

    def vacuum_database(self) -> None:
        """Optimize database file size"""
        with self.engine.connect() as conn:
            conn.execute(text("VACUUM"))

    def backup_database(self, backup_path: str) -> None:
        """Create database backup"""
        source = self.config.db_path / self.config.db_name
        if not source.exists():
            raise FileNotFoundError(f"Database file not found: {source}")
        shutil.copy2(source, backup_path)

    # === HEALTH CHECK API ===

    def health_check(self) -> dict[str, Any]:
        """Database health check"""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1")).fetchone()

            print("Database health: healthy")
            return {
                "status": "healthy",
                "database_info": self.database_info
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def close(self) -> None:
        """Close all connections"""
        self.engine.dispose()


# === GLOBAL DATABASE INSTANCE ===

# Global database manager instance
db = DatabaseManager(DatabaseConfig())


# Convenience functions for global access
def get_session():
    """Global session accessor"""
    return db.get_session()


def get_transaction():
    """Global transaction accessor"""
    return db.get_transaction()


def init_database():
    """Initialize database with all tables"""
    db.create_all_tables()
    return db.health_check()


if __name__ == "__main__":
    print("=== Database Initialization ===")
    info = init_database()
    print(f"Database: {info['database_info']['database_path']}")
    print(f"Database size: {info['database_info']['file_size_mb']} MB")
    print(f"Tables: {info['database_info']['tables']}")
