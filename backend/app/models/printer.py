"""
Pas de table printer en DB — l'état vient du MQTT en mémoire.
Ce fichier expose le dataclass partagé.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AMSTray:
    id: int
    color: str = ""
    filament_type: str = ""
    remain: int = 0
    uuid: str = ""
    info_idx: str = ""
    empty: bool = False
    spool_id: Optional[int] = None


@dataclass
class AMS:
    id: int
    trays: list[AMSTray] = field(default_factory=list)
    humidity: int = 0
    temp: float = 0.0


@dataclass
class PrinterState:
    connected: bool = False
    status: str = "IDLE"        # IDLE, RUNNING, PAUSE, FINISH, FAILED
    progress: float = 0.0
    layer: int = 0
    total_layers: int = 0
    remaining_minutes: int = 0
    nozzle_temp: float = 0.0
    target_nozzle_temp: float = 0.0
    bed_temp: float = 0.0
    target_bed_temp: float = 0.0
    chamber_temp: float = 0.0
    print_name: str = ""
    thumbnail: Optional[str] = None
    ams_list: list[AMS] = field(default_factory=list)
    job_id: str = ""
    use_ams: bool = True
