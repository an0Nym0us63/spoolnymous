import os
import sqlite3
from datetime import datetime, timedelta
import math
from collections import defaultdict
from config import COST_BY_HOUR

db_config = {"db_path": os.path.join(os.getcwd(), 'data', "3d_printer_logs.db")}

MAIN_COLOR_FAMILIES = {
    'Noir': (0, 0, 0),
    'Blanc': (255, 255, 255),
    'Gris': (160, 160, 160),
    'Rouge': (220, 20, 60),
    'Orange': (255, 140, 0),
    'Jaune': (255, 220, 0),
    'Vert': (80, 200, 120),
    'Bleu': (100, 150, 255),
    'Violet': (160, 32, 240),
    'Marron': (150, 75, 0)
}

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
                duration REAL,
                number_of_items INTEGER DEFAULT 1,
                group_id INTEGER,
                original_name TEXT,
                FOREIGN KEY (group_id) REFERENCES print_groups(id)
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

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS print_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                print_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                FOREIGN KEY (print_id) REFERENCES prints(id) ON DELETE CASCADE
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS print_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                number_of_items INTEGER DEFAULT 1,
                created_at TEXT
            )
        ''')

        conn.commit()
        conn.close()

    else:
        conn = sqlite3.connect(db_config["db_path"])
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(prints)")
        columns = [row[1] for row in cursor.fetchall()]
        if "number_of_items" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN number_of_items INTEGER DEFAULT 1")
        if "duration" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN duration REAL")
        if "group_id" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN group_id INTEGER REFERENCES print_groups(id)")
        if "original_name" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN original_name TEXT")
            cursor.execute("UPDATE prints SET original_name = file_name WHERE original_name IS NULL OR original_name = ''")

        cursor.execute("PRAGMA table_info(print_groups)")
        group_columns = [row[1] for row in cursor.fetchall()]
        if "number_of_items" not in group_columns:
            cursor.execute("ALTER TABLE print_groups ADD COLUMN number_of_items INTEGER DEFAULT 1")
        if "created_at" not in group_columns:
            cursor.execute("ALTER TABLE print_groups ADD COLUMN created_at TEXT")

            # initialiser created_at pour les groupes existants
            cursor.execute("SELECT id FROM print_groups")
            for row in cursor.fetchall():
                gid = row[0]
                cursor.execute("""
                    SELECT print_date FROM prints
                    WHERE group_id = ?
                    ORDER BY id DESC LIMIT 1
                """, (gid,))
                result = cursor.fetchone()
                if result and result[0]:
                    cursor.execute("""
                        UPDATE print_groups SET created_at = ? WHERE id = ?
                    """, (result[0], gid))
                else:
                    cursor.execute("""
                        UPDATE print_groups SET created_at = DATETIME('now') WHERE id = ?
                    """, (gid,))

        conn.commit()
        conn.close()


def insert_print(file_name: str, print_type: str, image_file: str = None, print_date: str = None, duration: float = 0) -> int:
    if print_date is None:
        print_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO prints (print_date, file_name, print_type, image_file, duration, original_name)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (print_date, file_name, print_type, image_file, duration, file_name))
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

def closest_family(hex_color: str) -> str:
    """
    Retourne la famille de couleur principale la plus proche.
    """
    distances = {
        famille: color_distance(hex_color, '#{:02X}{:02X}{:02X}'.format(*rgb))
        for famille, rgb in MAIN_COLOR_FAMILIES.items()
    }
    return min(distances.items(), key=lambda x: x[1])[0]


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
        words = [w.strip().lower() for w in search.split() if w.strip()]
        word_clauses = []
        for w in words:
            word_clauses.append("""
                LOWER(p.file_name) LIKE ?
                OR EXISTS (
                    SELECT 1 FROM print_tags pt
                    WHERE pt.print_id = p.id AND LOWER(pt.tag) LIKE ?
                )
                OR EXISTS (
                    SELECT 1 FROM print_groups pg
                    WHERE pg.id = p.group_id AND LOWER(pg.name) LIKE ?
                )
            """)
            params.extend([f"%{w}%"] * 3)

        if word_clauses:
            where_clauses.append(f"( {' OR '.join(word_clauses)} )")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Total count
    cursor.execute(f'''
        SELECT COUNT(DISTINCT p.id)
        FROM prints p
        LEFT JOIN filament_usage f ON f.print_id = p.id
        LEFT JOIN print_groups pg ON pg.id = p.group_id
        {where_sql}
    ''', params)
    total_count = cursor.fetchone()[0]

    # Load prints with optional LIMIT/OFFSET
    base_query = f'''
        SELECT DISTINCT p.id AS id,
            p.print_date,
            p.file_name,
            p.original_name,
            p.print_type,
            p.image_file,
            p.duration,
            p.number_of_items,
            pg.id AS group_id,
            pg.name AS group_name,
            pg.number_of_items AS group_number_of_items,
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
        LEFT JOIN print_groups pg ON pg.id = p.group_id
        {where_sql}
        ORDER BY p.print_date DESC
    '''


    if limit is not None:
        base_query += " LIMIT ? OFFSET ?"
        query_params = params + [limit, offset]
    else:
        query_params = params

    cursor.execute(base_query, query_params)
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

def update_print_history_field(print_id: int, field: str, value) -> None:
    """
    Met à jour un champ donné pour une impression de l'historique.
    """
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    query = f"UPDATE prints SET {field} = ? WHERE id = ?"
    cursor.execute(query, (value, print_id))
    conn.commit()
    conn.close()

def create_print_group(name: str) -> int:
    """
    Crée un groupe d'impressions et retourne son ID.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO print_groups (name, created_at) VALUES (?, ?)
    """, (name, now_str))
    group_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return group_id

def get_print_groups() -> list[dict]:
    """
    Retourne la liste des groupes existants.
    """
    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, name, created_at 
        FROM print_groups 
        ORDER BY created_at DESC
    """)
    groups = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return groups

def update_print_group_field(group_id: int, field: str, value) -> None:
    """
    Met à jour un champ donné pour un groupe d'impressions.
    """
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    query = f"UPDATE print_groups SET {field} = ? WHERE id = ?"
    cursor.execute(query, (value, group_id))
    conn.commit()
    conn.close()

def update_group_created_at(group_id: int) -> None:
    """
    Met à jour created_at d’un groupe avec la date du print le plus récent (id le plus élevé).
    Supprime le groupe s’il ne reste plus de prints.
    """
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("""
        SELECT print_date FROM prints
        WHERE group_id = ?
        ORDER BY id DESC LIMIT 1
    """, (group_id,))
    result = cursor.fetchone()
    if result:
        cursor.execute("""
            UPDATE print_groups SET created_at = ? WHERE id = ?
        """, (result[0], group_id))
    else:
        # plus aucun print dans le groupe → suppression du groupe
        cursor.execute("""
            DELETE FROM print_groups WHERE id = ?
        """, (group_id,))
    conn.commit()
    conn.close()

def get_group_id_of_print(print_id: int) -> int | None:
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("SELECT group_id FROM prints WHERE id = ?", (print_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def get_statistics(period: str = "all", filters: dict = None, search: str = None) -> dict:
    """
    Récupère des statistiques globales sur les impressions avec filtres optionnels.
    :param period: "all", "7d", "1m", "1y"
    :param filters: dictionnaire avec 'filament_type' et 'color' (valeurs multiples)
    :param search: chaîne libre (tag, nom de fichier ou groupe)
    """
    from spoolman_service import fetchSpools

    filters = filters or {}

    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Filtrage temporel
    date_clause = ""
    params = []
    now = datetime.now()

    if period == "day":
        since = now - timedelta(days=1)
    elif period == "7d":
        since = now - timedelta(days=7)
    elif period == "1m":
        since = now - timedelta(days=30)
    elif period == "1y":
        since = now - timedelta(days=365)
    else:
        since = None

    if since:
        date_clause = "p.print_date >= ?"
        params.append(since.strftime("%Y-%m-%d %H:%M:%S"))

    # Filtres filament_type
    if filters.get("filament_type"):
        placeholders = ",".join("?" for _ in filters["filament_type"])
        clause = f"f.filament_type IN ({placeholders})"
        params.extend(filters["filament_type"])
        date_clause = f"{date_clause} AND {clause}" if date_clause else clause

    # Filtres color (via familles)
    if filters.get("color"):
        cursor.execute("SELECT DISTINCT color FROM filament_usage WHERE color IS NOT NULL")
        all_colors = [row[0] for row in cursor.fetchall()]
        selected_hexes_by_family = []
        for fam in filters["color"]:
            hexes = [c for c in all_colors if fam in two_closest_families(c)]
            if hexes:
                selected_hexes_by_family.append(hexes)

        if selected_hexes_by_family:
            color_subclause = " OR ".join(
                ["f.color IN (" + ",".join("?" for _ in hexes) + ")" for hexes in selected_hexes_by_family]
            )
            date_clause = f"{date_clause} AND ({color_subclause})" if date_clause else f"({color_subclause})"
            for hexes in selected_hexes_by_family:
                params.extend(hexes)

    # Filtres recherche texte libre
    if search:
        words = [w.strip().lower() for w in search.split() if w.strip()]
        for w in words:
            date_clause += f""" AND (
                LOWER(p.file_name) LIKE ?
                OR EXISTS (
                    SELECT 1 FROM print_tags pt WHERE pt.print_id = p.id AND LOWER(pt.tag) LIKE ?
                )
            )"""
            params.extend([f"%{w}%"] * 2)

    where_sql = f"WHERE {date_clause}" if date_clause else ""

    # Charger les impressions
    cursor.execute(f"""
        SELECT DISTINCT p.id, p.duration
        FROM prints p
        LEFT JOIN filament_usage f ON f.print_id = p.id
        {where_sql}
    """, params)
    prints = cursor.fetchall()
    print_ids = [p["id"] for p in prints]

    if not prints:
        return {
            "total_prints": 0,
            "total_duration": 0.0,
            "total_weight": 0.0,
            "filament_cost": 0.0,
            "electric_cost": 0.0,
            "total_cost": 0.0
        }

    # Durée totale
    total_duration = sum(p["duration"] or 0 for p in prints)

    # Charger l’usage de filament
    cursor.execute(f"""
        SELECT print_id, spool_id, grams_used, filament_type, color
        FROM filament_usage
        WHERE print_id IN ({','.join('?' for _ in print_ids)})
    """, print_ids)
    usage = cursor.fetchall()
    conn.close()

    spools_by_id = {spool["id"]: spool for spool in fetchSpools(False, True)}

    total_weight = 0.0
    filament_cost = 0.0

    for u in usage:
        grams = u["grams_used"]
        spool = spools_by_id.get(u["spool_id"])
        cost_per_gram = spool.get("cost_per_gram", 0.0) if spool else 0.0
        total_weight += grams
        filament_cost += grams * cost_per_gram

    duration_hours = total_duration / 3600
    electric_cost = duration_hours * float(COST_BY_HOUR)

    vendor_counts = {}
    for u in usage:
        spool = spools_by_id.get(u["spool_id"])
        if spool:
            vendor = spool.get("filament", {}).get("vendor", {}).get("name")
            if vendor:
                vendor_counts[vendor] = vendor_counts.get(vendor, 0) + u["grams_used"]

    vendor_pie = {
        "labels": list(vendor_counts.keys()),
        "values": list(vendor_counts.values())
    }

    duration_bins = [0] * 11
    for p in prints:
        duration = p["duration"]
        if not duration or duration <= 0:
            continue
        h = duration / 3600
        index = min(int(h), 10)
        duration_bins[index] += 1

    duration_histogram = {
        "labels": [f"{i}–{i+1}h" for i in range(10)] + ["≥10h"],
        "values": duration_bins
    }

    filament_type_counts = {}
    for u in usage:
        ftype = u["filament_type"]
        if ftype:
            filament_type_counts[ftype] = filament_type_counts.get(ftype, 0) + u["grams_used"]

    filament_type_pie = {
        "labels": list(filament_type_counts.keys()),
        "values": list(filament_type_counts.values())
    }

    color_family_counts = {}
    color_family_colors = {}
    for u in usage:
        hex_color = u["color"]
        grams = u["grams_used"]
        if not hex_color:
            continue
        family = closest_family(hex_color)
        if family:
            color_family_counts[family] = color_family_counts.get(family, 0) + grams
            if family not in color_family_colors:
                rgb = MAIN_COLOR_FAMILIES[family]
                color_family_colors[family] = '#{:02X}{:02X}{:02X}'.format(*rgb)

    color_family_pie = {
        "labels": list(color_family_counts.keys()),
        "values": list(color_family_counts.values()),
        "colors": [color_family_colors[f] for f in color_family_counts.keys()]
    }

    return {
        "total_prints": len(print_ids),
        "total_duration": duration_hours,
        "total_weight": total_weight,
        "filament_cost": filament_cost,
        "electric_cost": electric_cost,
        "total_cost": filament_cost + electric_cost,
        "vendor_pie": vendor_pie,
        "duration_histogram": duration_histogram,
        "filament_type_pie": filament_type_pie,
        "color_family_pie": color_family_pie
    }

create_database()
