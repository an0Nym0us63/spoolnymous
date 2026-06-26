from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional
from ....db.session import get_db
from ....services.settings_service import get_all_settings, set_setting
from ....core.mqtt import mqtt_manager
from ....core.security import hash_password
from .auth import get_current_user

router = APIRouter()

EXPOSED_SETTINGS = [
    "PRINTER_IP", "PRINTER_ID", "PRINTER_ACCESS_CODE", "PRINTER_NAME",
    "ADMIN_USERNAME", "COST_BY_HOUR", "AUTO_SPEND",
]


class SettingsOut(BaseModel):
    PRINTER_IP: str = ""
    PRINTER_ID: str = ""
    PRINTER_ACCESS_CODE: str = ""
    PRINTER_NAME: str = ""
    ADMIN_USERNAME: str = "admin"
    COST_BY_HOUR: str = "0"
    AUTO_SPEND: str = "true"


class SettingsUpdate(BaseModel):
    PRINTER_IP: Optional[str] = None
    PRINTER_ID: Optional[str] = None
    PRINTER_ACCESS_CODE: Optional[str] = None
    PRINTER_NAME: Optional[str] = None
    ADMIN_USERNAME: Optional[str] = None
    ADMIN_PASSWORD: Optional[str] = None
    COST_BY_HOUR: Optional[str] = None
    AUTO_SPEND: Optional[str] = None


@router.get("", response_model=SettingsOut)
async def get_settings(db: AsyncSession = Depends(get_db), _: str = Depends(get_current_user)):
    all_s = await get_all_settings(db)
    return SettingsOut(
        PRINTER_IP=all_s.get("PRINTER_IP", ""),
        PRINTER_ID=all_s.get("PRINTER_ID", ""),
        PRINTER_ACCESS_CODE="***" if all_s.get("PRINTER_ACCESS_CODE") else "",
        PRINTER_NAME=all_s.get("PRINTER_NAME", ""),
        ADMIN_USERNAME=all_s.get("ADMIN_USERNAME", "admin"),
        COST_BY_HOUR=all_s.get("COST_BY_HOUR", "0"),
        AUTO_SPEND=all_s.get("AUTO_SPEND", "true"),
    )


@router.patch("")
async def update_settings(
    body: SettingsUpdate,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(get_current_user),
):
    printer_changed = False
    for field, value in body.model_dump(exclude_none=True).items():
        if field == "ADMIN_PASSWORD":
            await set_setting(db, "ADMIN_PASSWORD_HASH", hash_password(value))
        else:
            await set_setting(db, field, value)
        if field in ("PRINTER_IP", "PRINTER_ID", "PRINTER_ACCESS_CODE"):
            printer_changed = True

    if printer_changed:
        ip = (body.PRINTER_IP or "")
        pid = (body.PRINTER_ID or "")
        code = (body.PRINTER_ACCESS_CODE or "")
        if ip and pid and code:
            mqtt_manager.reconnect(ip, pid, code)

    return {"ok": True}
