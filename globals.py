from threading import Lock

PRINTER_STATUS = {
        "estimated_end": "-",
        "remaining_time_str": "-"
    }
PRINTER_STATUS_LOCK = Lock()