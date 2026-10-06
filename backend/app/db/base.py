"""Declarative base + a single import point that registers every ORM model
with SQLAlchemy's metadata, so Alembic autogeneration sees all tables."""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass



