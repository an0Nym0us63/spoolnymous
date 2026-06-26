from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column
from ..db.session import Base


class Setting(Base):
    """Paramètres persistés en DB (écrasent les env vars si présents)."""
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
