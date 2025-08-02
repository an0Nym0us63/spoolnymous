import os
import sqlite3
from datetime import datetime, timedelta
import math
from collections import defaultdict
import operator
from deep_translator import GoogleTranslator
import re
from config import COST_BY_HOUR as RAW_COST
COST_BY_HOUR = float(RAW_COST)

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

def sort_pie_data(pie):
    """
    Trie labels, values (et couleurs si présentes) par ordre décroissant des valeurs.
    """
    combined = list(zip(pie["labels"], pie["values"], pie.get("colors", [None] * len(pie["labels"]))))
    combined.sort(key=lambda x: x[1], reverse=True)

    sorted_labels, sorted_values, sorted_colors = zip(*combined)
    result = {
        "labels": list(sorted_labels),
        "values": list(sorted_values)
    }
    if pie.get("colors"):
        result["colors"] = list(sorted_colors)
    return result

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
                translated_name TEXT,
                status TEXT DEFAULT 'SUCCESS',
                status_note TEXT,
                sold_units INTEGER DEFAULT 0,
                sold_price_total REAL DEFAULT NULL,
                total_weight REAL DEFAULT 0.0,
                total_cost REAL DEFAULT 0.0,
                total_normal_cost REAL DEFAULT 0.0,
                electric_cost REAL DEFAULT 0.0,
                full_cost REAL DEFAULT 0.0,
                full_normal_cost REAL DEFAULT 0.0,
                full_cost_by_item REAL DEFAULT 0.0,
                full_normal_cost_by_item REAL DEFAULT 0.0,
                margin REAL DEFAULT 0.0,
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
                cost REAL DEFAULT 0.0,
                normal_cost REAL DEFAULT 0.0,
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
                created_at TEXT,
                sold_units INTEGER DEFAULT 0,
                sold_price_total REAL DEFAULT NULL
                primary_print_id INTEGER,
                total_weight REAL DEFAULT 0.0,
                total_cost REAL DEFAULT 0.0,
                total_normal_cost REAL DEFAULT 0.0,
                electric_cost REAL DEFAULT 0.0,
                full_cost REAL DEFAULT 0.0,
                full_normal_cost REAL DEFAULT 0.0,
                full_cost_by_item REAL DEFAULT 0.0,
                full_normal_cost_by_item REAL DEFAULT 0.0,
                margin REAL DEFAULT 0.0
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
        if "translated_name" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN translated_name TEXT")
            cursor.execute("SELECT id, file_name FROM prints")
            for pid, fname in cursor.fetchall():
                translated = update_translated_name(fname)
                cursor.execute("UPDATE prints SET translated_name = ? WHERE id = ?", (translated, pid))
        if "status" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN status TEXT DEFAULT 'SUCCESS'")
            cursor.execute("UPDATE prints SET status = 'SUCCESS' WHERE status IS NULL")
        if "status_note" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN status_note TEXT")
        if "sold_units" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN sold_units INTEGER DEFAULT 0")
        if "sold_price_total" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN sold_price_total REAL DEFAULT NULL")
        if "total_weight" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN total_weight REAL DEFAULT 0.0")
        if "total_cost" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN total_cost REAL DEFAULT 0.0")
        if "total_normal_cost" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN total_normal_cost REAL DEFAULT 0.0")
        if "electric_cost" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN electric_cost REAL DEFAULT 0.0")
        if "full_cost" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN full_cost REAL DEFAULT 0.0")
        if "full_normal_cost" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN full_normal_cost REAL DEFAULT 0.0")
        if "full_cost_by_item" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN full_cost_by_item REAL DEFAULT 0.0")
        if "full_normal_cost_by_item" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN full_normal_cost_by_item REAL DEFAULT 0.0")
        if "margin" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN margin REAL DEFAULT 0.0")
        
        cursor.execute("PRAGMA table_info(filament_usage)")
        columns = [col[1] for col in cursor.fetchall()]

        if "cost" not in columns:
            cursor.execute("ALTER TABLE filament_usage ADD COLUMN cost REAL DEFAULT 0.0")
        if "normal_cost" not in columns:
            cursor.execute("ALTER TABLE filament_usage ADD COLUMN normal_cost REAL DEFAULT 0.0")
            
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
        if "primary_print_id" not in group_columns:
            cursor.execute("ALTER TABLE print_groups ADD COLUMN primary_print_id INTEGER")
        if "sold_units" not in group_columns:
            cursor.execute("ALTER TABLE print_groups ADD COLUMN sold_units INTEGER DEFAULT 0")
        if "sold_price_total" not in group_columns:
            cursor.execute("ALTER TABLE print_groups ADD COLUMN sold_price_total REAL DEFAULT NULL")
        if "total_weight" not in group_columns:
            cursor.execute("ALTER TABLE print_groups ADD COLUMN total_weight REAL DEFAULT 0.0")
        if "total_cost" not in group_columns:
            cursor.execute("ALTER TABLE print_groups ADD COLUMN total_cost REAL DEFAULT 0.0")
        if "total_normal_cost" not in group_columns:
            cursor.execute("ALTER TABLE print_groups ADD COLUMN total_normal_cost REAL DEFAULT 0.0")
        if "electric_cost" not in group_columns:
            cursor.execute("ALTER TABLE print_groups ADD COLUMN electric_cost REAL DEFAULT 0.0")
        if "full_cost" not in group_columns:
            cursor.execute("ALTER TABLE print_groups ADD COLUMN full_cost REAL DEFAULT 0.0")
        if "full_normal_cost" not in group_columns:
            cursor.execute("ALTER TABLE print_groups ADD COLUMN full_normal_cost REAL DEFAULT 0.0")
        if "full_cost_by_item" not in group_columns:
            cursor.execute("ALTER TABLE print_groups ADD COLUMN full_cost_by_item REAL DEFAULT 0.0")
        if "full_normal_cost_by_item" not in group_columns:
            cursor.execute("ALTER TABLE print_groups ADD COLUMN full_normal_cost_by_item REAL DEFAULT 0.0")
        if "margin" not in group_columns:
            cursor.execute("ALTER TABLE print_groups ADD COLUMN margin REAL DEFAULT 0.0")

        conn.commit()
        conn.close()

def update_translated_name(name):
    source = "en"
    target = "fr"
    ctx_prefix = "__ctx__ "  # Contexte neutre pour forcer la traduction
    contextualized_text = f"This is a {name}"

    translated = GoogleTranslator(source=source, target=target).translate(contextualized_text)


    # Supprimer les préfixes de contexte traduits (insensibles à la casse)
    prefix_pattern = r"^(ceci est (un|une)|c'est (un|une)?|il s'agit d'(un|une)|il s'agit de)\s+"
    translated = re.sub(prefix_pattern, '', translated, flags=re.IGNORECASE)

    return translated.strip()

def clean_print_name(raw_name: str) -> str:
    # 1. Supprimer l'extension (.stl, .gcode, etc.)
    name, _ = os.path.splitext(raw_name)

    # 2. Remplacer uniquement les underscores `_` et les tirets longs `–` par des espaces
    name = name.replace('_', ' ')
    name = name.replace('–', ' ')  # tiret long (alt+0150 ou U+2013)

    # 3. Supprimer les suffixes techniques type "v2", "final3", "rev1"
    name = re.sub(r'\b(v|rev|final)\d*\b', '', name, flags=re.IGNORECASE)

    # 4. Réduire les espaces multiples
    name = re.sub(r'\s+', ' ', name)

    # 5. Nettoyage des bords + capitalisation intelligente
    name = name.strip()
    name = name[:1].upper() + name[1:] if name else name

    return name

def insert_print(file_name: str, print_type: str, image_file: str = None, print_date: str = None, duration: float = 0) -> int:
    if print_date is None:
        print_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cleaned = clean_print_name(file_name)
    translated = clean_print_name(update_translated_name(cleaned))

    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO prints (print_date, file_name, print_type, image_file, duration, original_name, translated_name)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (print_date, cleaned, print_type, image_file, duration, file_name, translated))
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

def two_closest_families(hex_color: str, threshold: float = 45.0) -> list[str]:
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
    from mqtt_bambulab import fetchSpools
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

    spools = fetchSpools(cached=False, archived=True)
    grouped = defaultdict(list)

    for s in spools:
        filament = s.get("filament")
        if not filament:
            continue
        parts = [
            filament.get("vendor", {}).get("name", ""),
            filament.get("material", ""),
            filament.get("name", "")
        ]
        display = " - ".join(part for part in parts if part)
        grouped[display].append(s)

    filaments = []
    for display_name, spools in grouped.items():
        ids = [str(s["id"]) for s in spools]
        raw_color = next(
                (s["filament"].get("color_hex") for s in spools if s.get("filament") and s["filament"].get("color_hex")),
                None
                )
        color = f"#{raw_color.lstrip('#')}" if raw_color else None
        filaments.append({
            "ids": ids,
            "display_name": display_name,
            "color": color
        })

    return {
        "filament_types": filament_types,
        "colors": sorted(families),
        "filaments": filaments
    }

def get_prints_with_filament(filters=None, search=None) -> list:
    filters = filters or {}
    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query_filters = []
    values = []

    if "filament_id" in filters:
        query_filters.append("f.spool_id = ?")
        values.append(filters["filament_id"])
    if "filament_type" in filters:
        query_filters.append("f.filament_type = ?")
        values.append(filters["filament_type"])
    if "family_color" in filters:
        query_filters.append("f.color LIKE ?")
        values.append(f"{filters['family_color']}%")
    if "status" in filters:
        query_filters.append("p.status = ?")
        values.append(filters["status"])

    if search:
        query_filters.append("(p.file_name LIKE ? OR p.translated_name LIKE ? OR pg.name LIKE ?)")
        values.extend([f"%{search}%"] * 3)

    where_clause = "WHERE " + " AND ".join(query_filters) if query_filters else ""

    query = f"""
        SELECT p.*, pg.name as group_name, pg.number_of_items as group_number_of_items
        FROM prints p
        LEFT JOIN print_groups pg ON p.group_id = pg.id
        {where_clause}
        ORDER BY p.print_date DESC
    """
    cursor.execute(query, values)
    prints = [dict(row) for row in cursor.fetchall()]

    for p in prints:
        cursor.execute("SELECT * FROM filament_usage WHERE print_id = ?", (p["id"],))
        p["filament_usage"] = [dict(u) for u in cursor.fetchall()]

    conn.close()
    return prints


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
    cleaned = clean_print_name(new_filename)
    translated = clean_print_name(update_translated_name(cleaned))
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE prints
        SET file_name = ?, translated_name = ?
        WHERE id = ?
    ''', (cleaned, translated, print_id))
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
            search_clause = f"""(
                LOWER(p.file_name) LIKE ?
                OR EXISTS (
                    SELECT 1 FROM print_tags pt WHERE pt.print_id = p.id AND LOWER(pt.tag) LIKE ?
                )
            )"""
            if date_clause:
                date_clause += f" AND {search_clause}"
            else:
                date_clause = search_clause
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
            "total_cost": 0.0,
            "vendor_pie": {"labels": [], "values": []},
            "duration_histogram": {"labels": [], "values": []},
            "filament_type_pie": {"labels": [], "values": []},
            "color_family_pie": {"labels": [], "values": [], "colors": []},
            "top_filaments": {"labels": [], "values": []}
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
    
    if filters.get("color"):
        selected_families = set(filters["color"])
        usage = [
            u for u in usage
            if u["color"] and any(f in selected_families for f in two_closest_families(u["color"]))
        ]

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
    
    filament_totals = {}
    for u in usage:
        spool = spools_by_id.get(u["spool_id"])
        if spool:
            vendor = spool.get("filament", {}).get("vendor", {}).get("name", "Inconnu")
            type_ = u["filament_type"] if "filament_type" in u.keys() else "Inconnu"
            name = spool.get("filament", {}).get("name", "Sans nom")
            key = f"{vendor} - {type_} - {name}"
            filament_totals[key] = filament_totals.get(key, 0.0) + u["grams_used"]
    
    sorted_filaments = sorted(filament_totals.items(), key=lambda x: x[1], reverse=True)[:15]
    top_filaments = {
        "labels": [label for label, _ in sorted_filaments],
        "values": [val for _, val in sorted_filaments]
    }
    
    stats_data = {
        "total_prints": len(print_ids),
        "total_duration": duration_hours,
        "total_weight": total_weight,
        "filament_cost": filament_cost,
        "electric_cost": electric_cost,
        "total_cost": filament_cost + electric_cost,
        "vendor_pie": vendor_pie,
        "duration_histogram": duration_histogram,
        "filament_type_pie": filament_type_pie,
        "color_family_pie": color_family_pie,
        "top_filaments": top_filaments
    }
    
    # Et maintenant que stats_data existe, tu peux faire tes tris :
    stats_data["vendor_pie"] = sort_pie_data(stats_data["vendor_pie"])
    stats_data["filament_type_pie"] = sort_pie_data(stats_data["filament_type_pie"])
    stats_data["color_family_pie"] = sort_pie_data(stats_data["color_family_pie"])
    # Réordonner les couleurs
    ordered_families = stats_data["color_family_pie"]["labels"]
    stats_data["color_family_pie"]["colors"] = [
        stats_data["color_family_pie"]["colors"][stats_data["color_family_pie"]["labels"].index(fam)]
        for fam in ordered_families
    ]
    conn.close()
    
    return stats_data

def adjustDuration(print_id: int, duration_seconds: int) -> None:
    """
    Met à jour la durée (en secondes) d’une impression dans l’historique.
    """
    update_print_history_field(print_id, "duration", duration_seconds)

def set_group_primary_print(group_id: int, print_id: int):
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("UPDATE print_groups SET primary_print_id = ? WHERE id = ?", (print_id, group_id))
    conn.commit()
    conn.close()

def set_sold_info(print_id: int, is_group: bool, total_price: float, sold_units: int) -> None:
    """
    Met à jour le nombre d’unités vendues et le total vendu pour un print ou un groupe.
    """
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()

    if is_group:
        cursor.execute("""
            UPDATE print_groups
            SET sold_units = ?, sold_price_total = ?
            WHERE id = ?
        """, (sold_units, total_price, print_id))
    else:
        cursor.execute("""
            UPDATE prints
            SET sold_units = ?, sold_price_total = ?
            WHERE id = ?
        """, (sold_units, total_price, print_id))

    conn.commit()
    conn.close()

def recalculate_filament_usage(usage_id: int, spools_by_id: dict) -> dict:
    """
    Recalcule et met à jour le coût réel et standardisé d'un filament_usage.

    :param usage_id: ID de l'entrée dans filament_usage
    :param spools_by_id: dictionnaire {spool_id: spool_data}, avec les clés:
                         - "cost_per_gram"
                         - "filament_cost_per_gram"
    :return: dict avec les valeurs recalculées {cost, normal_cost, grams_used}
    """
    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM filament_usage WHERE id = ?", (usage_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return {}

    grams_used = row["grams_used"]
    spool_id = row["spool_id"]
    spool = spools_by_id.get(spool_id)

    if not spool:
        conn.close()
        return {}

    cost_per_gram = spool.get("cost_per_gram", 0.0)
    normal_cost_per_gram = spool.get("filament_cost_per_gram", 0.0)

    cost = grams_used * cost_per_gram
    normal_cost = grams_used * normal_cost_per_gram

    cursor.execute("""
        UPDATE filament_usage
        SET cost = ?, normal_cost = ?
        WHERE id = ?
    """, (cost, normal_cost, usage_id))

    conn.commit()
    conn.close()

    return {
        "cost": cost,
        "normal_cost": normal_cost,
        "grams_used": grams_used
    }

def recalculate_print_data(print_id: int, spools_by_id: dict) -> None:
    """
    Recalcule et met à jour tous les champs de coûts pour un print donné.

    :param print_id: ID du print à recalculer
    :param spools_by_id: dictionnaire {spool_id: spool_data} venant de fetchSpools()
    """
    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Récupérer le print
    cursor.execute("SELECT * FROM prints WHERE id = ?", (print_id,))
    print_row = cursor.fetchone()
    if not print_row:
        conn.close()
        return

    duration = print_row["duration"] or 0.0
    number_of_items = print_row["number_of_items"] or 1
    sold_price_total = print_row["sold_price_total"] or 0.0
    sold_units = print_row["sold_units"] or 0

    # Récupérer les usages de filament
    cursor.execute("SELECT * FROM filament_usage WHERE print_id = ?", (print_id,))
    usages = cursor.fetchall()

    total_cost = total_normal_cost = total_weight = 0.0
    for usage in usages:
        result = recalculate_filament_usage(usage["id"], spools_by_id)
        total_cost += result.get("cost", 0.0)
        total_normal_cost += result.get("normal_cost", 0.0)
        total_weight += result.get("grams_used", 0.0)

    # Calculs électricité
    electric_cost = (duration / 3600.0) * COST_BY_HOUR if duration else 0.0

    full_cost = total_cost + electric_cost
    full_normal_cost = total_normal_cost + electric_cost
    full_cost_by_item = full_cost / number_of_items if number_of_items else 0.0
    full_normal_cost_by_item = full_normal_cost / number_of_items if number_of_items else 0.0
    margin = sold_price_total - (sold_units * full_cost_by_item)

    # Mise à jour dans prints
    cursor.execute("""
        UPDATE prints SET
            total_weight = ?, total_cost = ?, total_normal_cost = ?,
            electric_cost = ?, full_cost = ?, full_normal_cost = ?,
            full_cost_by_item = ?, full_normal_cost_by_item = ?, margin = ?
        WHERE id = ?
    """, (
        total_weight, total_cost, total_normal_cost,
        electric_cost, full_cost, full_normal_cost,
        full_cost_by_item, full_normal_cost_by_item, margin,
        print_id
    ))

    conn.commit()
    conn.close()

def recalculate_group_data(group_id: int, spools_by_id: dict) -> None:
    """
    Recalcule et met à jour tous les champs de coûts pour un groupe donné.

    :param group_id: ID du groupe
    :param spools_by_id: dictionnaire {spool_id: spool_data} venant de fetchSpools()
    """
    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Récupérer le groupe
    cursor.execute("SELECT * FROM print_groups WHERE id = ?", (group_id,))
    group = cursor.fetchone()
    if not group:
        conn.close()
        return

    number_of_items = group["number_of_items"] or 1
    sold_price_total = group["sold_price_total"] or 0.0
    sold_units = group["sold_units"] or 0

    # Récupérer tous les prints liés
    cursor.execute("SELECT id FROM prints WHERE group_id = ?", (group_id,))
    prints = cursor.fetchall()

    total_cost = total_normal_cost = total_weight = 0.0
    electric_cost = full_cost = full_normal_cost = 0.0

    for p in prints:
        recalculate_print_data(p["id"], spools_by_id)
        cursor.execute("SELECT * FROM prints WHERE id = ?", (p["id"],))
        print_row = cursor.fetchone()

        total_cost += print_row["total_cost"]
        total_normal_cost += print_row["total_normal_cost"]
        total_weight += print_row["total_weight"]
        electric_cost += print_row["electric_cost"]
        full_cost += print_row["full_cost"]
        full_normal_cost += print_row["full_normal_cost"]

    full_cost_by_item = full_cost / number_of_items if number_of_items else 0.0
    full_normal_cost_by_item = full_normal_cost / number_of_items if number_of_items else 0.0
    margin = sold_price_total - (sold_units * full_cost_by_item)

    # Mise à jour de la table print_groups
    cursor.execute("""
        UPDATE print_groups SET
            total_weight = ?, total_cost = ?, total_normal_cost = ?,
            electric_cost = ?, full_cost = ?, full_normal_cost = ?,
            full_cost_by_item = ?, full_normal_cost_by_item = ?, margin = ?
        WHERE id = ?
    """, (
        total_weight, total_cost, total_normal_cost,
        electric_cost, full_cost, full_normal_cost,
        full_cost_by_item, full_normal_cost_by_item, margin,
        group_id
    ))

    conn.commit()
    conn.close()

def cleanup_orphan_data() -> None:
    """
    Supprime les données orphelines :
    - filament_usage sans print associé
    - print_tags sans print associé
    - print_groups sans print (optionnel)
    """
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()

    # Filaments sans print
    cursor.execute("""
        DELETE FROM filament_usage
        WHERE print_id NOT IN (SELECT id FROM prints)
    """)

    # Tags sans print
    cursor.execute("""
        DELETE FROM print_tags
        WHERE print_id NOT IN (SELECT id FROM prints)
    """)

    # Groupes sans aucun print (optionnel)
    cursor.execute("""
        DELETE FROM print_groups
        WHERE id NOT IN (SELECT DISTINCT group_id FROM prints WHERE group_id IS NOT NULL)
    """)

    conn.commit()
    conn.close()

create_database()
