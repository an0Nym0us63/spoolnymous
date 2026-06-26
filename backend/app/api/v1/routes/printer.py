from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from ....core.mqtt import get_state, mqtt_manager
from ....models.printer import PrinterState
from .auth import get_current_user

router = APIRouter()


class TrayOut(BaseModel):
    id: int
    color: str
    filament_type: str
    remain: int
    uuid: str
    empty: bool
    spool_id: Optional[int] = None


class AMSOut(BaseModel):
    id: int
    trays: list[TrayOut]
    humidity: int
    temp: float


class PrinterStatusOut(BaseModel):
    connected: bool
    status: str
    progress: float
    layer: int
    total_layers: int
    remaining_minutes: int
    nozzle_temp: float
    target_nozzle_temp: float
    bed_temp: float
    target_bed_temp: float
    chamber_temp: float
    print_name: str
    thumbnail: Optional[str]
    ams_list: list[AMSOut]
    job_id: str


@router.get("/status", response_model=PrinterStatusOut)
async def printer_status(_: str = Depends(get_current_user)):
    s = get_state()
    return PrinterStatusOut(
        connected=s.connected,
        status=s.status,
        progress=s.progress,
        layer=s.layer,
        total_layers=s.total_layers,
        remaining_minutes=s.remaining_minutes,
        nozzle_temp=s.nozzle_temp,
        target_nozzle_temp=s.target_nozzle_temp,
        bed_temp=s.bed_temp,
        target_bed_temp=s.target_bed_temp,
        chamber_temp=s.chamber_temp,
        print_name=s.print_name,
        thumbnail=s.thumbnail,
        ams_list=[
            AMSOut(
                id=a.id,
                trays=[TrayOut(**t.__dict__) for t in a.trays],
                humidity=a.humidity,
                temp=a.temp,
            )
            for a in s.ams_list
        ],
        job_id=s.job_id,
    )
