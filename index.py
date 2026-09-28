"""Vercel entrypoint for the existing SmartBank FastAPI application."""

from src.api.main import app

__all__ = ["app"]
