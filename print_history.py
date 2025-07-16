import os
import sqlite3
from datetime import datetime
import math

db_config = {"db_path": os.path.join(os.getcwd(), 'data', "3d_printer_logs.db")}

COLOR_FAMILIES = {
    # Neutres
    'Black': (0, 0, 0),
    'White': (255, 255, 255),
    'Grey': (160, 160, 160),

    # Rouges et dérivés
    'Red': (220, 20, 60),         # Crimson
    'Dark Red': (139, 0, 0),      # sombre
    'Pink': (255, 182, 193),      # pastel
    'Magenta': (255, 0, 255),     # fuchsia
    'Brown': (150, 75, 0),        # chocolat

    # Jaunes et dérivés
    'Yellow': (255, 220, 0),      # chaud
    'Gold': (212, 175, 55),       # doré
    'Orange': (255, 140, 0),      # foncé

    # Verts
    'Green': (80, 200, 120),      # gazon
    'Dark Green': (0, 100, 0),    # forêt
    'Lime': (191, 255, 0),        # fluo
    'Teal': (0, 128, 128),        # turquoise

    # Bleus et violets
    'Blue': (100, 150, 255),      # clair
    'Navy': (0, 0, 128),          # foncé
    'Cyan': (0, 255, 255),        # turquoise clair
    'Lavender': (230, 230, 250),  # violet pastel
    'Purple': (160, 32, 240), 
    'Dark Purple': (90, 60, 120), # violet foncé
}


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
        
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS print_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                print_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                FOREIGN KEY (print_id) REFERENCES prints(id) ON DELETE CASCADE
            )
        ''')

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

def update_filament_usage(print_id, spool_id, new_grams_used):
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE filament_usage
        SET grams_used = ?
        WHERE print_id = ? AND spool_id = ?
    """, (new_grams_used, print_id, spool_id))
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

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')[:6]
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    r /= 255
    g /= 255
    b /= 255

    def gamma_correct(c):
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

    r = gamma_correct(r)
    g = gamma_correct(g)
    b = gamma_correct(b)

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    l = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return (l, a, b)

def color_distance(hex1: str, hex2: str) -> float:
    rgb1 = hex_to_rgb(hex1)
    rgb2 = hex_to_rgb(hex2)
    lab1 = rgb_to_lab(*rgb1)
    lab2 = rgb_to_lab(*rgb2)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))

def two_closest_families(hex_color: str, threshold: float = 60.0) -> list[str]:
    """
    Retourne la famille la plus proche et la deuxième si sa distance est < threshold.
    """
    distances = {
        family: color_distance(hex_color, '#{:02X}{:02X}{:02X}'.format(*rgb))
        for family, rgb in COLOR_FAMILIES.items()
    }
    sorted_families = sorted(distances.items(), key=lambda x: x[1])

    result = [sorted_families[0][0]]  # toujours la plus proche
    if sorted_families[1][1] <= threshold:
        result.append(sorted_families[1][0])
    return result


def get_distinct_values():
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT filament_type FROM filament_usage")
    filament_types = sorted([row[0] for row in cursor.fetchall()])
    cursor.execute("SELECT DISTINCT color FROM filament_usage WHERE color IS NOT NULL")
    raw_colors = [row[0] for row in cursor.fetchall()]
    families = set()
    for hex_color in raw_colors:
        families.update(two_closest_families(hex_color))
    conn.close()
    return {
        "filament_types": filament_types,
        "colors": sorted(families)
    }

def get_prints_with_filament(offset=0, limit=10, filters=None, search=None):
    filters = filters or {}
    where_clauses = []
    params = []

    if filters.get("filament_type"):
        placeholders = ",".join("?" for _ in filters["filament_type"])
        where_clauses.append(f"f.filament_type IN ({placeholders})")
        params.extend(filters["filament_type"])

    if filters.get("color"):
        color_families = filters["color"]
        conn = sqlite3.connect(db_config["db_path"])
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT color FROM filament_usage WHERE color IS NOT NULL")
        all_colors = [row[0] for row in cursor.fetchall()]
        conn.close()
    
        selected_hexes_by_family = []
        for fam in color_families:
            hexes = [
                c for c in all_colors
                if fam in two_closest_families(c)
            ]
            if hexes:
                selected_hexes_by_family.append(hexes)
    
        if selected_hexes_by_family:
            where_clauses.append(f"""
                p.id IN (
                    SELECT fu.print_id
                    FROM filament_usage fu
                    WHERE {" OR ".join(["fu.color IN (" + ",".join("?" for _ in hexes) + ")" for hexes in selected_hexes_by_family])}
                    GROUP BY fu.print_id
                    HAVING COUNT(DISTINCT fu.color) >= ?
                )
            """)
            for hexes in selected_hexes_by_family:
                params.extend(hexes)
            params.append(len(selected_hexes_by_family))

    if search:
        where_clauses.append("""
            (
                LOWER(p.file_name) LIKE ?
                OR EXISTS (
                    SELECT 1 FROM print_tags pt
                    WHERE pt.print_id = p.id AND LOWER(pt.tag) LIKE ?
                )
            )
        """)
        search_param = f"%{search.lower()}%"
        params.extend([search_param, search_param])

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

def update_print_filename(print_id: int, new_filename: str):
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE prints
        SET file_name = ?
        WHERE id = ?
    ''', (new_filename, print_id))
    conn.commit()
    conn.close()

def get_filament_for_print(print_id: int):
    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT spool_id, grams_used, filament_type, color
        FROM filament_usage
        WHERE print_id = ?
    ''', (print_id,))
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def delete_print(print_id: int):
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute('DELETE FROM prints WHERE id = ?', (print_id,))
    conn.commit()
    conn.close()
    
def get_tags_for_print(print_id: int):
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("SELECT tag FROM print_tags WHERE print_id = ?", (print_id,))
    tags = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tags

def add_tag_to_print(print_id: int, tag: str):
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("INSERT INTO print_tags (print_id, tag) VALUES (?, ?)", (print_id, tag))
    conn.commit()
    conn.close()

def remove_tag_from_print(print_id: int, tag: str):
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("DELETE FROM print_tags WHERE print_id = ? AND tag = ?", (print_id, tag))
    conn.commit()
    conn.close()


create_database()
