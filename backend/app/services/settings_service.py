from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from ..models.setting import Setting
from ..core.config import settings as app_settings
import os

_cache: dict[str, str] = {}


async def get_setting(db: AsyncSession, key: str, default: str = "") -> str:
    if key in _cache:
        return _cache[key]
    result = await db.execute(select(Setting).where(Setting.key == key))
    row = result.scalar_one_or_none()
    if row:
        _cache[key] = row.value
        return row.value
    # Fallback env
    return os.getenv(key, default)


async def set_setting(db: AsyncSession, key: str, value: str) -> None:
    result = await db.execute(select(Setting).where(Setting.key == key))
    row = result.scalar_one_or_none()
    if row:
        row.value = value
    else:
        db.add(Setting(key=key, value=value))
    await db.commit()
    _cache[key] = value


async def get_all_settings(db: AsyncSession) -> dict[str, str]:
    result = await db.execute(select(Setting))
    return {row.key: row.value for row in result.scalars().all()}


def invalidate_cache():
    _cache.clear()
