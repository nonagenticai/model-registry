"""
Alembic configuration for asynchronous migrations.

This file sets up an async engine for running migrations, using the
DATABASE_URL environment variable when available.
"""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

import models.db  # noqa: F401  # ensures models are imported for metadata
from db import Base

# Load logging configuration
config = context.config
try:
    if config.config_file_name is not None:
        fileConfig(config.config_file_name)
except KeyError:
    pass

# Use the metadata from our models
target_metadata = Base.metadata


def _resolve_db_url() -> str | None:
    """Pick the DB URL from the same env var the running service uses.

    Prefer MODEL_REGISTRY_DATABASE_URL (what the deployment sets) and fall back
    to DATABASE_URL for local / CI use. The fallback alembic.ini sqlalchemy.url
    only applies when neither env var is set.
    """
    return os.getenv("MODEL_REGISTRY_DATABASE_URL") or os.getenv("DATABASE_URL")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = _resolve_db_url() or config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section) or {}
    db_url = _resolve_db_url() or configuration.get("sqlalchemy.url")

    # Convert postgresql:// to postgresql+psycopg:// for async driver
    if db_url and db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

    configuration["sqlalchemy.url"] = db_url or ""

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
