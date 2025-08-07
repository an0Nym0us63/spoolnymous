import os
import sqlite3

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

init_settings_table()
