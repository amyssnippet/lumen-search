from logging.config import fileConfig
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context
from engine.db.models import Base
from engine.db.session import DATABASE_URL, engine

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = os.environ.get("DATABASE_URL") or config.get_main_option("sqlalchemy.url") or DATABASE_URL
    if url.startswith("driver://") or not url:
        url = DATABASE_URL
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    target_url = os.environ.get("DATABASE_URL") or config.get_main_option("sqlalchemy.url") or DATABASE_URL
    if target_url.startswith("driver://") or not target_url:
        target_url = DATABASE_URL
    if target_url.startswith("postgresql://"):
        target_url = target_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    if target_url and (target_url != DATABASE_URL or not engine):
        from sqlalchemy import create_engine
        is_sql = target_url.startswith("sqlite")
        connect_args = {"check_same_thread": False} if is_sql else {}
        connectable = create_engine(target_url, connect_args=connect_args)
    else:
        connectable = engine

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True if target_url.startswith("sqlite") else False
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

