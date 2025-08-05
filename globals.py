from threading import Lock
from time import time

PRINTER_STATUS = {
        "estimated_end": "-",
        "remaining_time_str": "-"
    }
PRINTER_STATUS_LOCK = Lock()

PROCESSED_JOBS = set()