import os
import sqlite3
from datetime import datetime, timedelta
import math
from collections import defaultdict
import operator
from deep_translator import GoogleTranslator
import re
import config
from config import get_app_setting,get_electric_rate_at
from camera import snapshot_to_print_file
from pathlib import Path
import re


import logging

logger = logging.getLogger(__name__)

db_config = {"db_path": os.path.join(os.getcwd(), 'data', "3d_printer_logs.db")}

PHOTO_SEQ_RE = re.compile(r'photo-(\d+)', re.IGNORECASE)

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

_IMPRESSION_RE = re.compile(r"^Impression[-_\s]*(\d+)%?$", re.IGNORECASE)

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
                job_id INTEGER DEFAULT 0,
                original_duration REAL,
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
                sold_price_total REAL DEFAULT NULL,
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tray_spool_map (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tray_uuid TEXT NOT NULL,
                tray_info_idx TEXT NOT NULL,
                color TEXT NOT NULL,
                spool_id INTEGER NOT NULL
            )
        ''')
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prints_file_name       ON prints(file_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prints_translated_name ON prints(translated_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prints_original_name   ON prints(original_name)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_print_tags_print_id    ON print_tags(print_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_print_tags_tag         ON print_tags(tag)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_print_groups_name      ON print_groups(name)")

        conn.commit()
        conn.close()

    else:
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
                job_id INTEGER DEFAULT 0,
                original_duration REAL,
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
                sold_price_total REAL DEFAULT NULL,
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tray_spool_map (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tray_uuid TEXT NOT NULL,
                tray_info_idx TEXT NOT NULL,
                color TEXT NOT NULL,
                spool_id INTEGER NOT NULL
            )
        ''')
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
        if "original_duration" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN original_duration REAL")
            cursor.execute("UPDATE prints SET original_duration = duration WHERE original_duration IS NULL")
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
        if "job_id" not in columns:
            cursor.execute("ALTER TABLE prints ADD COLUMN job_id INTEGER DEFAULT 0")
        
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prints_file_name       ON prints(file_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prints_translated_name ON prints(translated_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prints_original_name   ON prints(original_name)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_print_tags_print_id    ON print_tags(print_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_print_tags_tag         ON print_tags(tag)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_print_groups_name      ON print_groups(name)")
        conn.commit()
        conn.close()


def update_translated_name(name: str) -> str:
    if not name.strip():
        return ""

    # Traduction mot à mot
    forced_input = '\n'.join(name.split())
    mot_a_mot = GoogleTranslator(source='auto', target='fr').translate(forced_input)
    mot_a_mot = ' '.join(mot_a_mot.split('\n')).strip()

    # Traduction contextuelle
    contextuel = GoogleTranslator(source='auto', target='fr').translate(name.strip()).strip()

    if contextuel.lower() == name.lower():
        return contextuel

    # Dédoublonnage insensible à la casse
    seen = set()
    dedup_words = []
    for word in (mot_a_mot + ' ' + contextuel).split():
        key = word.lower()
        if key not in seen:
            seen.add(key)
            dedup_words.append(word)

    return ' '.join(dedup_words)

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

def insert_print(file_name: str, print_type: str, image_file: str = None, print_date: str = None, duration: float = 0, job_id: int = 0) -> int:
    if print_date is None:
        print_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cleaned = clean_print_name(file_name)
    translated = clean_print_name(update_translated_name(cleaned))

    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    status = "IN_PROGRESS"
    if (print_type == "manual"):
        status = "SUCCESS"
    cursor.execute('''
        INSERT INTO prints (print_date, file_name, print_type, image_file, duration, original_name, translated_name, job_id, status,original_duration)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (print_date, cleaned, print_type, image_file, duration, file_name, translated, job_id, status, duration))
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
    trigger_cost_recalculation(print_id)

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
    trigger_cost_recalculation(print_id)

def get_print_id_by_job_id(job_id: str | int) -> int | None:
    """
    Retourne l'ID du print lié à un job_id BambuLab.
    - Si plusieurs prints existent pour ce job_id, renvoie le plus récent.
    - Retourne None si introuvable.
    """
    if not job_id:
        return None
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    row = cursor.execute(
        "SELECT id FROM prints WHERE job_id = ? ORDER BY id DESC LIMIT 1",
        (str(job_id),)
    ).fetchone()

    return int(row[0]) if row else None

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
    trigger_cost_recalculation(print_id)

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
    from mqtt_bambulab import fetch_spools
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

    spools = fetch_spools(archived=True)
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
    having_clauses = []
    having_values = []

    # --- FILTRES CACHÉS: lien direct depuis Objets -------------------------
    # Priorité au ref_print_id si les deux sont fournis.
    if filters.get("__ref_print_id"):
        query_filters.append("p.id = ?")
        values.append(int(filters["__ref_print_id"]))
    elif filters.get("__ref_group_id"):
        query_filters.append("p.group_id = ?")
        values.append(int(filters["__ref_group_id"]))

    # --- filament_id: au moins un usage sur une des spools demandées -------
    if filters.get("filament_id"):
        ids = [v.strip() for v in filters["filament_id"] if v and v.strip()]
        if ids:
            placeholders = ",".join(["?"] * len(ids))
            query_filters.append(f"""
                EXISTS (
                    SELECT 1
                    FROM filament_usage fu
                    WHERE fu.print_id = p.id
                      AND fu.spool_id IN ({placeholders})
                )
            """)
            values.extend(ids)

    # --- filament_type: simple égalité sur l’alias du LEFT JOIN -----------
    if filters.get("filament_type"):
        query_filters.append("f.filament_type = ?")
        values.append(filters["filament_type"][0])

    # --- status: IN dynamique ---------------------------------------------
    if filters.get("status"):
        statuses = filters["status"]
        query_filters.append(f"p.status IN ({','.join(['?'] * len(statuses))})")
        values.extend(statuses)

    # --- color: HAVING sur agrégat ----------------------------------------
    if filters.get("color"):
        cursor.execute("SELECT DISTINCT color FROM filament_usage WHERE color IS NOT NULL")
        all_colors = [row[0] for row in cursor.fetchall()]
        for fam in filters["color"]:
            hexes = [c for c in all_colors if fam in two_closest_families(c)]
            if hexes:
                having_clauses.append(
                    "SUM(CASE WHEN f.color IN ({}) THEN 1 ELSE 0 END) > 0".format(",".join(["?"] * len(hexes)))
                )
                having_values.extend(hexes)

    # --- recherche plein-texte --------------------------------------------
    if search:
        words = [w.strip().lower() for w in search.split() if w.strip()]
        for w in words:
            query_filters.append("""
                (
                    LOWER(p.file_name) LIKE ?
                    OR LOWER(p.translated_name) LIKE ?
                    OR EXISTS (
                        SELECT 1 FROM print_tags pt
                        WHERE pt.print_id = p.id AND LOWER(pt.tag) LIKE ?
                    )
                    OR EXISTS (
                        SELECT 1 FROM print_groups pg2
                        WHERE pg2.id = p.group_id AND LOWER(pg2.name) LIKE ?
                    )
                )
            """)
            values.extend([f"%{w}%" ] * 4)

    where_clause = "WHERE " + " AND ".join(query_filters) if query_filters else ""
    group_by_clause = "GROUP BY p.id"
    having_clause = f"HAVING {' AND '.join(having_clauses)}" if having_clauses else ""

    query = f"""
        SELECT p.*, pg.name as group_name, pg.number_of_items as group_number_of_items,
               pg.primary_print_id as group_primary_print_id
        FROM prints p
        LEFT JOIN print_groups pg ON p.group_id = pg.id
        LEFT JOIN filament_usage f ON p.id = f.print_id
        {where_clause}
        {group_by_clause}
        {having_clause}
        ORDER BY p.print_date DESC
    """

    cursor.execute(query, values + having_values)
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

def update_print_field_with_job_id(job_id: int, field: str, value) -> None:
    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if field == "status":
        try:
            cursor.execute("""
                SELECT print_date,status, duration, original_duration
                FROM prints
                WHERE job_id = ?
                LIMIT 1
            """, (job_id,))
            row = cursor.fetchone()

            if row and row["print_date"]:
                print_datetime = datetime.strptime(row["print_date"], "%Y-%m-%d %H:%M:%S")
                new_duration = (datetime.now() - print_datetime).total_seconds()

                # --- Conditions de mise à jour de 'duration' ---
                can_update_duration = False

                # 1) duration ne doit pas avoir divergé de original_duration
                same_as_original = (row["duration"] == row["original_duration"])

                if same_as_original:
                    old = row["duration"]

                    # 2) La nouvelle durée doit être dans ±20% de l'ancienne, si old est exploitable
                    if isinstance(old, (int, float)) and old is not None and old > 0:
                        lower = 0.001 * float(old)
                        upper = 1.3 * float(old)
                        if lower <= new_duration <= upper:
                            can_update_duration = True
                    else:
                        # Pas de baseline fiable -> on n'applique pas la contrainte des 20%
                        can_update_duration = True

                if can_update_duration:
                    cursor.execute("""
                        UPDATE prints
                        SET duration = ?
                        WHERE job_id = ?
                    """, (new_duration, job_id))
                    if row["status"] == 'IN_PROGRESS':
                        cursor.execute("""
                            UPDATE prints
                            SET status = ?
                            WHERE job_id = ?
                        """, (value, job_id))
                else:
                    if row["status"] == 'IN_PROGRESS':
                        # On ne touche pas à duration : on met à jour uniquement le statut
                        cursor.execute("""
                            UPDATE prints
                            SET status = ?
                            WHERE job_id = ?
                        """, (value, job_id))
            else:
                # Pas de print_date -> mise à jour uniquement du statut
                cursor.execute("""
                    UPDATE prints
                    SET status = ?
                    WHERE job_id = ?
                """, (value, job_id))
        except Exception as e:
            print(f"[WARN] Erreur lors du calcul de durée pour job_id={job_id} : {e}")
            cursor.execute("""
                UPDATE prints
                SET status = ?
                WHERE job_id = ?
            """, (value, job_id))

    else:
        # Autres champs : mise à jour directe (⚠️ valider 'field' côté appelant pour éviter l'injection SQL)
        query = f"UPDATE prints SET {field} = ? WHERE job_id = ?"
        cursor.execute(query, (value, job_id))

    conn.commit()
    conn.close()

def delete_print(print_id: int):
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("SELECT group_id FROM prints WHERE id = ?", (print_id,))
    result = cursor.fetchone()
    group_id = result[0] if result else None

    # Suppression du print
    cursor.execute("DELETE FROM prints WHERE id = ?", (print_id,))
    conn.commit()
    conn.close()

    # Recalcul éventuel du groupe
    if group_id is not None:
        trigger_cost_recalculation(group_id, is_group=True)
    
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

# --- TAGS : batch & groupes -----------------------------------------------

def get_tags_for_prints(print_ids: list[int]) -> dict[int, list[str]]:
    """Retourne un mapping {print_id: [tags]} pour un ensemble de prints."""
    if not print_ids:
        return {}
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    q_marks = ",".join("?" for _ in print_ids)
    cursor.execute(f"SELECT print_id, tag FROM print_tags WHERE print_id IN ({q_marks})", tuple(print_ids))
    tags_by_print = defaultdict(list)
    for pid, tag in cursor.fetchall():
        tags_by_print[pid].append(tag)
    conn.close()
    return tags_by_print

def get_group_print_ids(group_id: int) -> list[int]:
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM prints WHERE group_id = ?", (group_id,))
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ids

def get_tags_for_group(group_id: int) -> list[str]:
    """Union des tags de tous les prints du groupe."""
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT pt.tag
        FROM print_tags pt
        JOIN prints p ON p.id = pt.print_id
        WHERE p.group_id = ?
    """, (group_id,))
    tags = sorted([r[0] for r in cursor.fetchall()], key=lambda s: s.lower())
    conn.close()
    return tags

def add_tag_to_group(group_id: int, tag: str) -> None:
    """Ajoute un tag à tous les prints du groupe (sans doublon)."""
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    # Insère pour les prints du groupe qui ne l'ont pas déjà
    cursor.execute("""
        INSERT INTO print_tags (print_id, tag)
        SELECT p.id, ?
        FROM prints p
        WHERE p.group_id = ?
          AND NOT EXISTS (
              SELECT 1 FROM print_tags pt WHERE pt.print_id = p.id AND pt.tag = ?
          )
    """, (tag, group_id, tag))
    conn.commit()
    conn.close()

def remove_tag_from_group(group_id: int, tag: str) -> None:
    """Supprime un tag de tous les prints du groupe."""
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM print_tags
        WHERE tag = ?
          AND print_id IN (SELECT id FROM prints WHERE group_id = ?)
    """, (tag, group_id))
    conn.commit()
    conn.close()

def update_print_history_field(print_id: int, field: str, value) -> None:
    """
    Met à jour un champ donné pour une impression de l'historique.
    """
    group_id = get_group_id_of_print(print_id)
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    query = f"UPDATE prints SET {field} = ? WHERE id = ?"
    cursor.execute(query, (value, print_id))
    conn.commit()
    conn.close()
    trigger_cost_recalculation(print_id)
    if group_id:
        trigger_cost_recalculation(group_id, is_group=True)
        

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
    Retourne la liste complète des groupes existants avec tous leurs champs.
    """
    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT *
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
    trigger_cost_recalculation(group_id, is_group=True)

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
    from filaments import fetch_spools

    filters = filters or {}

    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    now = datetime.now()
    since = None
    base_clauses = []
    usage_clauses = []
    params = []
    usage_params = []

    if period == "day":
        since = now - timedelta(days=1)
    elif period == "7d":
        since = now - timedelta(days=7)
    elif period == "1m":
        since = now - timedelta(days=30)
    elif period == "1y":
        since = now - timedelta(days=365)

    if since:
        base_clauses.append("p.print_date >= ?")
        params.append(since.strftime("%Y-%m-%d %H:%M:%S"))

    if search:
        words = [w.strip().lower() for w in search.split() if w.strip()]
        for w in words:
            base_clauses.append("""(
                LOWER(p.file_name) LIKE ?
                OR EXISTS (
                    SELECT 1 FROM print_tags pt WHERE pt.print_id = p.id AND LOWER(pt.tag) LIKE ?
                )
            )""")
            params.extend([f"%{w}%"] * 2)

    if filters.get("filament_type"):
        placeholders = ",".join("?" for _ in filters["filament_type"])
        usage_clauses.append(f"f.filament_type IN ({placeholders})")
        usage_params.extend(filters["filament_type"])

    if filters.get("color"):
        cursor.execute("SELECT DISTINCT color FROM filament_usage WHERE color IS NOT NULL")
        all_colors = [row[0] for row in cursor.fetchall()]
        selected_hexes_by_family = []
        for fam in filters["color"]:
            hexes = [c for c in all_colors if fam in two_closest_families(c)]
            if hexes:
                selected_hexes_by_family.append(hexes)

        if selected_hexes_by_family:
            color_subclause = " OR ".join([
                "f.color IN (" + ",".join("?" for _ in hexes) + ")" for hexes in selected_hexes_by_family
            ])
            usage_clauses.append(f"({color_subclause})")
            for hexes in selected_hexes_by_family:
                usage_params.extend(hexes)

    base_where = f"WHERE {' AND '.join(base_clauses)}" if base_clauses else ""
    usage_where = f"WHERE {' AND '.join(base_clauses + usage_clauses)}" if base_clauses or usage_clauses else ""
    full_params = params + usage_params

    cursor.execute(f"""
        SELECT DISTINCT p.id, p.file_name, p.duration, p.group_id
        FROM prints p
        LEFT JOIN filament_usage f ON f.print_id = p.id
        {usage_where}
    """, full_params)
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
            "top_filaments": {"labels": [], "values": []},
            "top_longest_prints": [],
            "top_heaviest_prints": [],
            "total_margin": 0.0
        }

    cursor.execute(f"""
        SELECT COALESCE(SUM(margin), 0) AS margin_sum
        FROM prints p
        {base_where}
    """, params)
    print_margin = cursor.fetchone()["margin_sum"] or 0

    group_id_clause = f"{base_where} AND group_id IS NOT NULL" if base_where else "WHERE group_id IS NOT NULL"
    cursor.execute(f"""
        SELECT DISTINCT group_id FROM prints p
        {group_id_clause}
    """, params)
    group_ids = [row["group_id"] for row in cursor.fetchall() if row["group_id"] is not None]

    group_margin = 0
    if group_ids:
        placeholders = ",".join("?" for _ in group_ids)
        cursor.execute(f"""
            SELECT COALESCE(SUM(margin), 0) AS margin_sum FROM print_groups
            WHERE id IN ({placeholders})
        """, group_ids)
        group_margin = cursor.fetchone()["margin_sum"] or 0

    total_margin = print_margin + group_margin

    total_duration = sum(p["duration"] or 0 for p in prints)

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

    spools_by_id = {spool["id"]: spool for spool in fetch_spools(archived=True)}

    total_weight = 0.0
    filament_cost = 0.0

    for u in usage:
        grams = u["grams_used"]
        spool = spools_by_id.get(u["spool_id"])
        cost_per_gram = spool.get("cost_per_gram", 0.0) if spool else 0.0
        total_weight += grams
        filament_cost += grams * cost_per_gram

    duration_hours = total_duration / 3600
    cursor.execute(
        f"SELECT COALESCE(SUM(electric_cost),0) FROM prints WHERE id IN ({','.join('?' for _ in print_ids)})",
        print_ids
    )
    electric_cost = float(cursor.fetchone()[0] or 0.0)

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
        "labels": [f"{i}–{i+1}h" for i in range(10)] + ["\u226510h"],
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

    sorted_filaments = sorted(filament_totals.items(), key=lambda x: x[1], reverse=True)[:20]
    top_filaments = {
        "labels": [label for label, _ in sorted_filaments],
        "values": [val for _, val in sorted_filaments]
    }

    durations_prints = [
        {"id": p["id"], "name": p["file_name"], "duration": (p["duration"] or 0) / 3600, "is_group": False}
        for p in prints if p["duration"]
    ]

    cursor.execute(f"""
        SELECT p.group_id AS id, MIN(pg.name) AS name, SUM(p.duration) AS duration
        FROM prints p
        JOIN print_groups pg ON p.group_id = pg.id
        WHERE p.id IN ({','.join('?' for _ in print_ids)}) AND p.group_id IS NOT NULL
        GROUP BY p.group_id
    """, print_ids)
    durations_groups = [
        {"id": r["id"], "name": r["name"], "duration": r["duration"] / 3600, "is_group": True}
        for r in cursor.fetchall() if r["duration"]
    ]

    cursor.execute(f"""
        SELECT p.id, p.file_name, SUM(fu.grams_used) AS total_grams
        FROM filament_usage fu
        JOIN prints p ON fu.print_id = p.id
        WHERE p.id IN ({','.join('?' for _ in print_ids)})
        GROUP BY p.id
    """, print_ids)
    weights_prints = [
        {"id": r["id"], "name": r["file_name"], "weight": r["total_grams"], "is_group": False}
        for r in cursor.fetchall()
    ]

    cursor.execute(f"""
        SELECT p.group_id AS id, MIN(pg.name) AS name, SUM(fu.grams_used) AS total_weight
        FROM prints p
        JOIN print_groups pg ON p.group_id = pg.id
        JOIN filament_usage fu ON fu.print_id = p.id
        WHERE p.id IN ({','.join('?' for _ in print_ids)}) AND p.group_id IS NOT NULL
        GROUP BY p.group_id
    """, print_ids)
    weights_groups = [
        {"id": r["id"], "name": r["name"], "weight": r["total_weight"], "is_group": True}
        for r in cursor.fetchall()
    ]

    stats_data = {
        "total_prints": len(print_ids),
        "total_duration": duration_hours,
        "total_weight": total_weight,
        "filament_cost": filament_cost,
        "electric_cost": electric_cost,
        "total_cost": filament_cost + electric_cost,
        "vendor_pie": sort_pie_data(vendor_pie),
        "duration_histogram": duration_histogram,
        "filament_type_pie": sort_pie_data(filament_type_pie),
        "color_family_pie": sort_pie_data(color_family_pie),
        "top_filaments": top_filaments,
        "total_margin": total_margin,
        "top_longest_prints": sorted(durations_prints + durations_groups, key=lambda x: x["duration"], reverse=True)[:30],
        "top_heaviest_prints": sorted(weights_prints + weights_groups, key=lambda x: x["weight"], reverse=True)[:30]
    }

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
    trigger_cost_recalculation(print_id, is_group)

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
    dt = None
    try:
        if print_row["print_date"]:
            dt = datetime.strptime(print_row["print_date"], "%Y-%m-%d %H:%M:%S")
    except Exception:
        dt = None
    rate = get_electric_rate_at(dt)
    electric_cost = ((duration or 0.0) / 3600.0) * rate

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

def trigger_cost_recalculation(target_id: int, is_group: bool = False) -> None:
    """
    Déclenche un recalcul du coût pour un print ou un groupe donné.
    """
    
    from filaments import fetch_spools
    spools_by_id = {spool["id"]: spool for spool in fetch_spools(archived=True)}

    if is_group:
        recalculate_group_data(target_id, spools_by_id)
    else:
        # Vérifie si le print appartient à un groupe
        conn = sqlite3.connect(db_config["db_path"])
        cursor = conn.cursor()
        cursor.execute("SELECT group_id FROM prints WHERE id = ?", (target_id,))
        result = cursor.fetchone()
        conn.close()

        group_id = result[0] if result else None

        if group_id is not None:
            recalculate_group_data(group_id, spools_by_id)
        else:
            recalculate_print_data(target_id, spools_by_id)

def get_latest_print():
    try:
        conn = sqlite3.connect(db_config["db_path"])
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(prints)")
        columns = [col[1] for col in cursor.fetchall()]

        cursor.execute("SELECT * FROM prints ORDER BY print_date DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(zip(columns, row))
        return None
    except Exception as e:
        print(f"[ERROR] get_latest_print: {e}")
        return None

tray_spool_map_cache = None

def load_tray_spool_map_cache(force_reload=False):
    global tray_spool_map_cache
    if tray_spool_map_cache is None or force_reload:
        conn = sqlite3.connect(db_config["db_path"])
        cursor = conn.cursor()
        cursor.execute("SELECT id, tray_uuid, tray_info_idx, color, spool_id FROM tray_spool_map")
        tray_spool_map_cache = cursor.fetchall()
        conn.close()

def get_tray_spool_map(tray_uuid: str, tray_info_idx: str, color: str) -> int | None:
    load_tray_spool_map_cache()
    base_color = color[:7].lower()
    for row in tray_spool_map_cache:
        _, uuid, info_idx, col, spool_id = row
        if uuid == tray_uuid and info_idx == tray_info_idx and color[:7].lower() == base_color:
            return spool_id
    return None

def set_tray_spool_map(tray_uuid: str, tray_info_idx: str, color: str, spool_id: int) -> None:
    global tray_spool_map_cache
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id FROM tray_spool_map
        WHERE tray_uuid = ? AND tray_info_idx = ? AND color = ?
    """, (tray_uuid, tray_info_idx, color))
    result = cursor.fetchone()

    if result:
        cursor.execute("""
            UPDATE tray_spool_map
            SET spool_id = ?
            WHERE id = ?
        """, (spool_id, result[0]))
    else:
        cursor.execute("""
            INSERT INTO tray_spool_map (tray_uuid, tray_info_idx, color, spool_id)
            VALUES (?, ?, ?, ?)
        """, (tray_uuid, tray_info_idx, color, spool_id))
    conn.commit()
    conn.close()
    tray_spool_map_cache = None  # Invalider le cache

def delete_tray_spool_map_by_id(map_id: int) -> None:
    global tray_spool_map_cache
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tray_spool_map WHERE id = ?", (map_id,))
    conn.commit()
    conn.close()
    tray_spool_map_cache = None  # Invalider le cache

def get_all_tray_spool_mappings() -> list[dict]:
    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tray_spool_map")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_all_tray_spool_mappings() -> None:
    global tray_spool_map_cache
    conn = sqlite3.connect(db_config["db_path"])
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tray_spool_map")
    conn.commit()
    conn.close()
    tray_spool_map_cache = None  # Invalider le cache


def snapshot_milestone(job_id: str, pct: int, basename: str | None = None) -> None:
    """
    Déclenche un snapshot pour un milestone (50/99/100) d'un job donné.
    - Résout print_id à partir de job_id (via la fonction *existante* du projet).
    - Si le fichier existe déjà (ex. après reboot), ne fait rien.
    - Sinon enregistre via camera.snapshot_to_print_file(print_id, basename).
    """
    # 1) Résoudre print_id via ta fonction existante
    try:
        # ✎✎✎ REMPLACE le nom par la fonction réelle de ton code :
        print_id = get_print_id_by_job_id(job_id)  # <-- EXISTANTE DANS TON PROJET
    except NameError:
        # Si le nom diffère, crée un alias/adapter ici.
        raise RuntimeError("Fonction 'get_print_id_by_job_id(job_id)' introuvable dans print_history.")

    if not print_id:
        logger.debug("snapshot_milestone: aucun print lié au job_id=%s", job_id)
        return

    # 2) Préparer le nom de fichier
    if not basename:
        basename = f"Impression-{pct}"

    # 3) Vérifier si le fichier existe déjà (utile au reboot)
    base_dir = Path(__file__).resolve().parent
    target = base_dir / "static" / "uploads" / "prints" / str(print_id) / f"{basename}.jpg"
    if target.exists():
        logger.debug("snapshot_milestone: déjà présent: %s", target)
        return

    # 4) Assurer un app_context pour que camera lise la config (PRINTER_IP/PRINTER_ACCESS_CODE)
    try:
        snapshot_to_print_file(print_id, basename)
        logger.info("Snapshot milestone %s (job_id=%s, print_id=%s) OK", basename, job_id, print_id)
    except Exception as e:
        logger.warning("Snapshot milestone %s (job_id=%s, print_id=%s) ÉCHEC: %s", basename, job_id, print_id, e)
        raise

def list_print_images(print_id: str | int | None = None):
    """
    Retourne une liste de dicts {url, name} des images trouvées pour ce print.
    - Priorité aux fichiers dont le nom commence par 'Impression XX%'
      (triés par XX décroissant)
    - Puis les autres (triés alphabétiquement).
    """
    base_dir = Path(__file__).resolve().parent
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    results = []

    def scan(dir_id):
        if dir_id is None:
            return
        d = base_dir / "static" / "uploads" / "prints" / str(dir_id)
        if not d.exists():
            return
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                results.append({
                    "url": f"/static/uploads/prints/{dir_id}/{p.name}",
                    "name": p.stem,
                })

    scan(print_id)

    # dédoublonnage
    seen = set()
    uniq = []
    for r in results:
        key = (r["url"], r["name"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    # tri personnalisé
    def sort_key(item):
        m = _IMPRESSION_RE.match(item["name"])
        if m:
            pct = int(m.group(1))
            return (0, -pct)   # groupe 0 = impressions, tri décroissant
        else:
            return (1, item["name"].lower())  # groupe 1 = autres, tri alphabétique

    return sorted(uniq, key=sort_key)

def list_group_images(group_id: str | int | None = None):
    """
    Retourne une liste de dicts {url, name} pour un groupe, en concaténant :
      1) Images propres au groupe:                static/uploads/groups/<group_id>/
      2) Images 'Photo-*' des prints du groupe    (via list_print_images), name préfixé 'Print-'
      3) Images 'Impression XX%' des prints       (via list_print_images), name préfixé 'Print-'

    Ordre strict : (1) groupe, (2) prints Photo-*, (3) prints Impression XX%.
    Dédoublonnage global par (url, name).
    """
    base_dir = Path(__file__).resolve().parent
    exts = {".jpg", ".jpeg", ".png", ".webp"}

    if group_id is None:
        return []

    # --- 1) Images du groupe ---
    group_results = []
    d = base_dir / "static" / "uploads" / "groups" / str(group_id)
    if d.exists():
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                group_results.append({
                    "url": f"/static/uploads/groups/{group_id}/{p.name}",
                    "name": p.stem,
                })

    # Tri interne des images du groupe :
    #  - Impression XX% d'abord (XX décroissant), puis alpha
    def sort_key_group(item):
        m = _IMPRESSION_RE.match(item["name"])
        if m:
            pct = int(m.group(1))
            return (0, -pct)
        return (1, item["name"].lower())

    group_results = sorted(group_results, key=sort_key_group)

    # --- 2 & 3) Images des prints du groupe ---
    photos_results = []      # uniquement noms commençant par 'Photo-'
    progress_results = []    # uniquement 'Impression XX%'

    try:
        print_ids = get_group_print_ids(int(group_id))
    except Exception:
        print_ids = []

    for pid in print_ids:
        for img in (list_print_images(pid) or []):
            name = img.get("name") or ""
            if _IMPRESSION_RE.match(name):
                progress_results.append({
                    "url": img["url"],
                    "name": f"Print-{name}",
                })
            elif name.lower().startswith("photo-"):
                photos_results.append({
                    "url": img["url"],
                    "name": f"Print-{name}",
                })
            else:
                # Si tu préfères ignorer les autres, commente la ligne suivante.
                photos_results.append({  # on les place dans le bucket "Photo-*" par défaut
                    "url": img["url"],
                    "name": f"Print-{name}",
                })

    # Tri interne des buckets prints :
    #  - 'Photo-*' tri alpha
    photos_results = sorted(photos_results, key=lambda x: (x["name"] or "").lower())
    #  - 'Impression XX%' tri par XX décroissant
    def sort_key_progress(item):
        m = _IMPRESSION_RE.match(item["name"].removeprefix("Print-"))
        return (0, -int(m.group(1))) if m else (1, item["name"].lower())
    progress_results = sorted(progress_results, key=sort_key_progress)

    # --- Dédoublonnage global et concat ordre strict ---
    seen = set()
    out = []

    for coll in (group_results, photos_results, progress_results):
        for r in coll:
            key = (r["url"], r["name"])
            if key not in seen:
                seen.add(key)
                out.append(r)

    return out

def _seq_from_name(name: str) -> int:
    """
    Extrait le numéro à partir de 'Photo-<N>.*'.
    Retourne un grand nombre si pas de numéro (pour être trié en dernier).
    """
    m = PHOTO_SEQ_RE.search(name or '')
    return int(m.group(1)) if m else 10**9

def _item_title(entity: str, entity_id: int) -> str:
    """
    Récupère un titre humain pour l’item :
    - print  : file_name ou 'Print #<id>'
    - group  : label/nom du groupe ou 'Groupe #<id>'
    Robuste    : tente plusieurs tables, fallback si manquantes.
    """
    
    logger.debug(entity + ' ' + str(entity_id))
    title = None
    try:
        conn = sqlite3.connect(db_config["db_path"])
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if entity == "prints":
            # essaie colonnes probables
            cur.execute("""
                SELECT file_name as name
                FROM prints
                WHERE id = ?
            """, (entity_id,))
            row = cur.fetchone()
            if row and row["name"]:
                title = row["name"]
        elif entity == "groups":
            # essaie print_groups d’abord, puis groups
            cur.execute("SELECT name as name FROM print_groups WHERE id = ?", (entity_id,))
            row = cur.fetchone()
            if row and row["name"]:
                title = row["name"]
    except Exception:
        pass
    logger.debug(title)
    if title and title.strip():
        return title.strip()
    return ("Print #{}".format(entity_id) if entity == "prints" else "Groupe #{}".format(entity_id))

def list_all_photos(prefix="Photo-", q: str | None = None):
    """
    Scanne /static/uploads/{prints,groups}/<id> et retourne la liste
    de toutes les photos dont le nom commence par `prefix` (par défaut Photo-).
    Tri par item : Photo-<N> ascendant.
    Renvoie une liste aplatie triée par groupe (ordre: récence de l’item).

    Si `q` est fourni (texte libre, case-insensitive) :
      - PRINTS  retenus si q ∈ file_name OR translated_name OR original_name OR tag
      - GROUPS  retenus si q ∈ name (print_groups.name puis fallback groups.name)
      - On ne renvoie que les photos des items retenus.
    """
    base_dir = Path(__file__).resolve().parent

    groups = defaultdict(list)  # key -> photos: { "prints:12": [ {...}, ... ], "groups:3": [...] }
    meta   = {}                 # key -> {'entity','entity_id','item_title','latest_mtime'}

    # 1) Scan FS et collecte des clés (prints/groups ayant au moins une photo)
    for entity in ("prints", "groups"):
        base = base_dir / "static" / "uploads" / entity
        if not base.exists():
            continue

        for item_dir in base.iterdir():
            if not item_dir.is_dir():
                continue
            try:
                entity_id = int(item_dir.name)
            except ValueError:
                continue

            key = f"{entity}:{entity_id}"
            latest_mtime = meta.get(key, {}).get("latest_mtime", 0)

            for f in item_dir.iterdir():
                if not f.is_file():
                    continue
                name = f.name
                if not name.lower().startswith(prefix.lower()):
                    continue
                # sécurité simple
                if name.lower().endswith(".3mf"):
                    continue

                try:
                    mtime = f.stat().st_mtime
                except Exception:
                    mtime = 0
                latest_mtime = max(latest_mtime, mtime)

                groups[key].append({
                    "entity": entity,
                    "entity_id": entity_id,
                    "url": f"/static/uploads/{entity}/{entity_id}/{name}",
                    "name": name,
                    "base_name": Path(name).stem,  # sans extension
                    "seq": _seq_from_name(name),   # pour tri asc
                })

            # Si aucune photo retenue, ne crée pas de meta pour cette clé
            if groups.get(key):
                meta[key] = {
                    "entity": entity,
                    "entity_id": entity_id,
                    "item_title": _item_title(entity, entity_id),  # existant
                    "latest_mtime": latest_mtime,
                }

    # 2) Filtrage optionnel par q (texte libre, case-insensitive)
    if q:
        q_norm = q.casefold()
        print_ids  = [m["entity_id"] for k, m in meta.items() if m["entity"] == "prints"]
        group_ids  = [m["entity_id"] for k, m in meta.items() if m["entity"] == "groups"]

        prints_meta = {}
        groups_meta = {}

        try:
            conn = sqlite3.connect(db_config["db_path"])
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # PRINTS: file_name / translated_name / original_name
            if print_ids:
                q_marks = ",".join("?" for _ in print_ids)
                cur.execute(f"""
                    SELECT id, file_name, translated_name, original_name
                    FROM prints
                    WHERE id IN ({q_marks})
                """, tuple(print_ids))
                for r in cur.fetchall():
                    prints_meta[r["id"]] = {
                        "file_name":       (r["file_name"]       or ""),
                        "translated_name": (r["translated_name"] or ""),
                        "original_name":   (r["original_name"]   or ""),
                        "tags": [],  # complété juste après
                    }

                # Tags associés (bulk)
                from collections import defaultdict as _dd
                if print_ids:
                    q_marks = ",".join("?" for _ in print_ids)
                    cur.execute(f"""
                        SELECT print_id, tag
                        FROM print_tags
                        WHERE print_id IN ({q_marks})
                    """, tuple(print_ids))
                    for pid, tag in cur.fetchall():
                        prints_meta.setdefault(pid, {"file_name":"", "translated_name":"", "original_name":"", "tags":[]})
                        if tag:
                            prints_meta[pid]["tags"].append(str(tag))

            # GROUPS: name (print_groups en priorité, puis groups)
            if group_ids:
                q_marks = ",".join("?" for _ in group_ids)
                cur.execute(f"SELECT id, name FROM print_groups WHERE id IN ({q_marks})", tuple(group_ids))
                tmp = {r["id"]: (r["name"] or "") for r in cur.fetchall()}
                missing = [gid for gid in group_ids if gid not in tmp]
                if missing:
                    q2 = ",".join("?" for _ in missing)
                    cur.execute(f"SELECT id, name FROM groups WHERE id IN ({q2})", tuple(missing))
                    for r in cur.fetchall():
                        tmp[r["id"]] = (r["name"] or "")
                groups_meta = tmp

        except Exception:
            # En cas de pépin DB, on ne filtre pas (comportement grâce)
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

        # Applique le filtre q sur les clés
        keep_keys = []
        for key, m in meta.items():
            if m["entity"] == "prints":
                pid = m["entity_id"]
                pm  = prints_meta.get(pid, {})
                hay = " ".join([
                    pm.get("file_name", ""),
                    pm.get("translated_name", ""),
                    pm.get("original_name", ""),
                    *pm.get("tags", []),
                ]).casefold()
                if q_norm in hay:
                    keep_keys.append(key)
            else:
                gid = m["entity_id"]
                name = (groups_meta.get(gid) or "").casefold()
                if q_norm in name:
                    keep_keys.append(key)

        # Filtre les groupes/métadonnées aux seules clés retenues
        groups = {k: groups[k] for k in keep_keys if k in groups}
        meta   = {k: meta[k]   for k in keep_keys if k in meta}

    # 3) Tri intra-groupe par seq asc, puis nom
    for key, photos in groups.items():
        photos.sort(key=lambda x: (x["seq"], x["name"]))

    # 4) Ordre des groupes : plus récents d’abord
    ordered_keys = sorted(meta.keys(), key=lambda k: meta[k]["latest_mtime"], reverse=True)

    # 5) Aplatit avec champs enrichis (schema inchangé)
    out = []
    for key in ordered_keys:
        title = meta[key]["item_title"]
        for ph in groups.get(key, []):
            out.append({
                "entity": ph["entity"],
                "entity_id": ph["entity_id"],
                "url": ph["url"],
                "name": ph["name"],             # ex: Photo-12.jpg
                "base_name": ph["base_name"],   # ex: Photo-12
                "item_title": title,            # ex: Mon Vase ou Groupe Foo
            })

    return out

create_database()
