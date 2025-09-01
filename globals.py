from threading import Lock,RLock
import copy
import time

PRINTER_STATUS = {
        "estimated_end": "-",
        "remaining_time_str": "-"
    }
PRINTER_STATUS_LOCK = Lock()

PROCESSED_JOBS = set()

PENDING_JOBS = {}

# PRINTER_STATUS existe déjà chez toi ; on garde le même objet
try:
    PRINTER_STATUS  # déjà défini
except NameError:
    PRINTER_STATUS = {}

# Verrou pour maj thread-safe (MQTT + web/routes)
_PR_LOCK = RLock()

def _deep_update(dst: dict, src: dict) -> None:
    """Fusionne src -> dst récursivement, sans écraser les sous-dicts existants."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v

def deep_merge(dst: dict, src: dict) -> None:
    """API publique si d'autres modules veulent réutiliser le merge profond."""
    _deep_update(dst, src)

def current_status_snapshot() -> dict:
    """Copie immuable de l'état courant, pour calculs/milestones."""
    with _PR_LOCK:
        return copy.deepcopy(PRINTER_STATUS)

def update_status(partial: dict) -> None:
    """
    Fusionne un delta (frame incomplète) dans l'état global.
    NE PAS y passer tout l'objet complet si la source ne l’a pas !
    """
    if not isinstance(partial, dict):
        return
    with _PR_LOCK:
        _deep_update(PRINTER_STATUS, partial)
        PRINTER_STATUS["_ts"] = time.time()