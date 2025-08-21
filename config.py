import os
import sqlite3
from datetime import datetime
import json as _json

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin"
EXTERNAL_SPOOL_AMS_ID = 255 # don't change
EXTERNAL_SPOOL_ID = 254 #  don't change
AUTO_SPEND = True
SPOOL_SORTING = os.getenv('SPOOL_SORTING', "filament.material:asc,filament.vendor.name:asc,filament.name:asc")
PRINTER_NAME=""

db_config = {"db_path": os.path.join(os.getcwd(), 'data', "3d_printer_logs.db")}

def init_settings_table():
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

app_settings_cache = None

def load_app_settings_cache(force_reload=False):
    global app_settings_cache
    if app_settings_cache is None or force_reload:
        conn = sqlite3.connect(db_config["db_path"])
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM settings")
        app_settings_cache = dict(cursor.fetchall())
        conn.close()

def get_app_setting(key: str, default=None, use_env=True) -> str | None:
    load_app_settings_cache()
    if key in app_settings_cache:
        return app_settings_cache[key]
    if use_env:
        return os.getenv(key, default)
    return default

def set_app_setting(key: str, value: str) -> None:
    global app_settings_cache
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO settings (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
    """, (key, value))
    conn.commit()
    conn.close()
    app_settings_cache = None  # Invalider le cache

def get_all_app_settings() -> dict:
    load_app_settings_cache()
    return app_settings_cache.copy()

def _parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    s = str(s).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s.replace("Z",""))
    except Exception:
        return None

def get_electric_tariffs() -> list[dict]:
    """
    Retourne une liste triée:
      [{"start": datetime|None, "price_per_hour": float}, ...]
    Lit settings["ELECTRICITY_TARIFFS"] (JSON). Fallback sur COST_BY_HOUR si vide.
    """
    settings = get_all_app_settings()
    raw = (settings.get("ELECTRICITY_TARIFFS") or "").strip()
    tariffs: list[dict] = []
    if raw:
        try:
            arr = _json.loads(raw)
            if isinstance(arr, list):
                for it in arr:
                    if not isinstance(it, dict): 
                        continue
                    start = _parse_dt(it.get("start") or it.get("start_at") or it.get("date") or "")
                    try:
                        price = float(it.get("price_per_hour") or it.get("hourly") or it.get("price") or it.get("value"))
                    except Exception:
                        continue
                    tariffs.append({"start": start, "price_per_hour": price})
        except Exception:
            pass
    if not tariffs:
        # héritage: coût fixe si aucun barème défini
        legacy = settings.get("COST_BY_HOUR", "")
        try:
            if str(legacy).strip() != "":
                tariffs.append({"start": None, "price_per_hour": float(legacy)})
        except Exception:
            pass
    tariffs.sort(key=lambda x: (x["start"] or datetime.min))
    return tariffs

def get_electric_rate_at(dt: datetime | None) -> float:
    """
    Retourne le prix/h effectif à la date dt.
    Règle: on prend le dernier tarif dont start <= dt.
    Si dt=None: on prend le DERNIER tarif (le plus récent).
    """
    tariffs = get_electric_tariffs()
    if not tariffs:
        return 0.0
    if dt is None:
        return float(tariffs[-1]["price_per_hour"])
    rate = float(tariffs[0]["price_per_hour"])
    for t in tariffs:
        st = t["start"]
        if st is None or st <= dt:
            rate = float(t["price_per_hour"])
        else:
            break
    return rate

init_settings_table()
