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
    s = s.strip().replace("T", " ")
    # on accepte "YYYY-MM-DD HH:MM[:SS]" ou juste "YYYY-MM-DD"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    # tentative isoformat standard
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None

def get_electric_tariffs() -> list[dict]:
    """
    Retourne une liste triée de tarifs sous la forme :
      [{"start": "YYYY-MM-DD HH:MM", "price_per_hour": 0.20}, ...]
    - Lit settings["ELECTRICITY_TARIFFS"] (JSON).
    - Fallback : si absent, on utilise COST_BY_HOUR (unique, sans start).
    """
    settings = get_all_app_settings()
    raw = (settings.get("ELECTRICITY_TARIFFS") or "").strip()
    tariffs: list[dict] = []

    if raw:
        try:
            arr = _json.loads(raw)
            if isinstance(arr, list):
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    start = item.get("start") or item.get("start_at") or item.get("date")
                    price = item.get("price_per_hour") or item.get("hourly") or item.get("price") or item.get("value")
                    if price is None:
                        continue
                    dt = _parse_dt(str(start)) if start else None
                    try:
                        p = float(price)
                    except Exception:
                        continue
                    tariffs.append({"start": dt, "price_per_hour": p})
        except Exception:
            # JSON invalide => on ignore silencieusement (l’UI prévient côté page)
            pass

    # Fallback legacy : COST_BY_HOUR si rien défini
    if not tariffs:
        legacy = settings.get("COST_BY_HOUR", "")
        if str(legacy).strip() != "":
            try:
                p = float(legacy)
                tariffs.append({"start": None, "price_per_hour": p})
            except Exception:
                pass

    # Trier par start croissant (None en premier)
    tariffs.sort(key=lambda x: (x["start"] or datetime.min))
    return tariffs

def get_electric_rate_at(dt: datetime | None) -> float:
    """
    Donne le prix/h effectif à l’instant dt.
    - Prend le dernier tarif dont start <= dt (ou le dernier défini si dt est None).
    - Retourne 0.0 si aucun tarif valide.
    """
    tariffs = get_electric_tariffs()
    if not tariffs:
        return 0.0
    if dt is None:
        return tariffs[-1]["price_per_hour"]  # dernier tarif

    rate = 0.0
    for t in tariffs:
        st = t["start"]
        if st is None or st <= dt:
            rate = t["price_per_hour"]
        else:
            break
    return rate

init_settings_table()
