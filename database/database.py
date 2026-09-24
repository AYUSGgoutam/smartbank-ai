"""SQLAlchemy setup with SQLite defaults and PostgreSQL URL support."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from dotenv import load_dotenv

load_dotenv()


class Base(DeclarativeBase):
    """Declarative base for application tables."""


def database_url() -> str:
    """Read the URL from the environment; create a local data folder by default."""
    configured = os.getenv("DATABASE_URL")
    if configured:
        return configured
    Path("data").mkdir(parents=True, exist_ok=True)
    return "sqlite:///./data/smartbank.db"


def make_engine(url: str | None = None):
    """Create an engine with SQLite thread access enabled for FastAPI workers."""
    target = url or database_url()
    options = {"check_same_thread": False} if target.startswith("sqlite") else {}
    return create_engine(target, connect_args=options, pool_pre_ping=True)


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    """Yield a request-scoped SQLAlchemy session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
