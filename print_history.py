import os
import sqlite3
from datetime import datetime


db_config = {"db_path": os.path.join(os.getcwd(), 'data', "3d_printer_logs.db")}


def create_database() -> None:
    if not os.path.exists(db_config["db_path"]):
        conn = sqlite3.connect(db_config["db_path"])
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                print_date TEXT NOT NULL,
                file_name TEXT NOT NULL,
                print_type TEXT NOT NULL,
                image_file TEXT,
                duration REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS filament_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                print_id INTEGER NOT NULL,
                spool_id INTEGER,
                filament_type TEXT NOT NULL,
                color TEXT NOT NULL,
                grams_used REAL NOT NULL,
                ams_slot INTEGER NOT NULL,
                FOREIGN KEY (print_id) REFERENCES prints (id) ON DELETE CASCADE
            )
        ''')

        conn.commit()
        conn.close()
    else:
        conn = sqlite3.connect(db_config["db_path"])
        cursor = conn.cursor()

        try:
            cursor.execute('ALTER TABLE prints ADD COLUMN duration REAL')
        except:
            pass

        conn.commit()
        conn.close()


def insert_print(file_name: str, print_type: str, image_file: str = None, print_date: str = None, duration: float = 0) -> int:
    if print_date is None:
        print_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO prints (print_date, file_name, print_type, image_file, duration)
        VALUES (?, ?, ?, ?, ?)
    ''', (print_date, file_name, print_type, image_file, duration))
    print_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return print_id


def insert_filament_usage(print_id: int, filament_type: str, color: str, grams_used: float, ams_slot: int) -> None:
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO filament_usage (print_id, filament_type, color, grams_used, ams_slot)
        VALUES (?, ?, ?, ?, ?)
    ''', (print_id, filament_type, color, grams_used, ams_slot))
    conn.commit()
    conn.close()


def update_filament_spool(print_id: int, filament_id: int, spool_id: int) -> None:
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE filament_usage
        SET spool_id = ?
        WHERE ams_slot = ? AND print_id = ?
    ''', (spool_id, filament_id, print_id))
    conn.commit()
    conn.close()


def get_prints_with_filament(offset=0, limit=10, filters=None):
    filters = filters or {}
    where_clauses = []
    params = []

    if filters.get("filament_type"):
        placeholders = ",".join("?" for _ in filters["filament_type"])
        where_clauses.append(f"f.filament_type IN ({placeholders})")
        params.extend(filters["filament_type"])

    if filters.get("print_type"):
        placeholders = ",".join("?" for _ in filters["print_type"])
        where_clauses.append(f"p.print_type IN ({placeholders})")
        params.extend(filters["print_type"])

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(f'''
        SELECT COUNT(DISTINCT p.id)
        FROM prints p
        LEFT JOIN filament_usage f ON f.print_id = p.id
        {where_sql}
    ''', params)
    total_count = cursor.fetchone()[0]

    cursor.execute(f'''
        SELECT DISTINCT p.id AS id, p.print_date AS print_date, p.file_name AS file_name, 
               p.print_type AS print_type, p.image_file AS image_file, p.duration AS duration,
               (
                   SELECT json_group_array(json_object(
                       'spool_id', f2.spool_id,
                       'filament_type', f2.filament_type,
                       'color', f2.color,
                       'grams_used', f2.grams_used,
                       'ams_slot', f2.ams_slot
                   )) FROM filament_usage f2 WHERE f2.print_id = p.id
               ) AS filament_info
        FROM prints p
        LEFT JOIN filament_usage f ON f.print_id = p.id
        {where_sql}
        ORDER BY p.print_date DESC
        LIMIT ? OFFSET ?
    ''', params + [limit, offset])
    prints = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return total_count, prints


def get_distinct_values():
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT filament_type FROM filament_usage")
    filament_types = sorted([row[0] for row in cursor.fetchall()])
    cursor.execute("SELECT DISTINCT print_type FROM prints")
    print_types = sorted([row[0] for row in cursor.fetchall()])
    conn.close()
    return {
        "filament_types": filament_types,
        "print_types": print_types,
    }


def get_filament_for_slot(print_id: int, ams_slot: int):
    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM filament_usage
        WHERE print_id = ? AND ams_slot = ?
    ''', (print_id, ams_slot))
    result = cursor.fetchone()
    conn.close()
    return result


create_database()
