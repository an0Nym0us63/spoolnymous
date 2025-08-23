import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from config import EXTERNAL_SPOOL_AMS_ID, EXTERNAL_SPOOL_ID,get_app_setting

from zoneinfo import ZoneInfo
import json
import urllib.parse
import urllib.request


# On réutilise la config DB telle qu'elle existe déjà dans le projet
from print_history import db_config, update_filament_spool

import logging
logger = logging.getLogger(__name__)
# ----------------------------------------------------------------------------
# Connexion & transactions
# ----------------------------------------------------------------------------

def trayUid(ams_id, tray_id):
  return f"{get_app_setting("PRINTER_ID","")}_{ams_id}_{tray_id}"

def _connect() -> sqlite3.Connection:
    """Retourne une connexion SQLite avec row_factory et clés étrangères actives."""
    conn = sqlite3.connect(db_config["db_path"])  # même DB que le reste du projet
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def _tx() -> Iterable[sqlite3.Cursor]:
    """Contexte transactionnel pratique.

    Usage:
        with _tx() as cur:
            cur.execute(...)
            ...
    """
    conn = _connect()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# Schéma (filaments & bobines)
# ----------------------------------------------------------------------------


def _column_exists(cur: sqlite3.Cursor, table: str, column: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == column for r in cur.fetchall())


def ensure_schema() -> None:
    """
    Crée/Met à jour le schéma minimal pour gérer filaments et bobines.

    Tables créées:
      - filaments: méta d'un modèle de filament (fabricant, matière, couleur, poids, etc.)
      - bobines  : instances physiques rattachées à un filament (poids restant, localisation, etc.)

    Notes de design:
      - Les dates sont stockées en ISO (TEXT) avec défaut `datetime('now')` côté SQLite.
      - `colors_array` est un TEXT (on peut y stocker du JSON ou une liste CSV).
      - `reference_id` et `external_filament_id` permettent de mapper avec des sources externes (ex: Spoolman).
      - Suppression d'un filament => suppression en cascade de ses bobines.
    """
    with _tx() as cur:
        # -----------------------------------------------------------------
        # Table FILAMENTS (modèle/catalogue)
        # -----------------------------------------------------------------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS filaments (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at           TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at           TEXT    NOT NULL DEFAULT (datetime('now')),

                name                 TEXT    NOT NULL,            -- nom commercial
                manufacturer         TEXT,                        -- fabricant/marque

                -- Gestion de couleur
                color                TEXT,                        -- couleur principale (hex ou nom)
                multicolor_type      TEXT,                        -- 'none', 'coaxial', 'gradient', etc.
                colors_array         TEXT,                        -- CSV de hex ex: "0047BB,BB22A3"

                material             TEXT,                        -- PLA, PETG, ABS, TPU, ...

                price                REAL,                        -- prix de référence (peut être surchargé par bobine)
                filament_weight_g    REAL,                        -- masse de filament utile (g)
                spool_weight_g       REAL,                        -- tare de la bobine (g)

                comment              TEXT,

                external_filament_id TEXT,                        -- identifiant externe (ex: Spoolman filament.id)
                reference_id         TEXT,                        -- ref libre interne/externe
                profile_id           TEXT                         -- ex: extra.filament_id (profil matériau)
            )
            """
        )

        # Index utiles sur filaments
        cur.execute("CREATE INDEX IF NOT EXISTS idx_filaments_name ON filaments(name)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_filaments_manu_mat ON filaments(manufacturer, material)"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_filaments_ext ON filaments(external_filament_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_filaments_profile ON filaments(profile_id)")

        # Trigger de mise à jour du timestamp
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_filaments_updated_at
            AFTER UPDATE ON filaments
            BEGIN
                UPDATE filaments SET updated_at = datetime('now') WHERE id = NEW.id;
            END;
            """
        )

        # -----------------------------------------------------------------
        # Table BOBINES (instances physiques)
        # -----------------------------------------------------------------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bobines (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                filament_id        INTEGER NOT NULL,

                created_at         TEXT    NOT NULL DEFAULT (datetime('now')),
                first_used_at      TEXT,                          -- date de première utilisation
                last_used_at       TEXT,                          -- date de dernière utilisation
                updated_at         TEXT    NOT NULL DEFAULT (datetime('now')),

                price_override     REAL,                          -- si NULL => fallback sur filaments.price
                remaining_weight_g REAL,                          -- poids restant (g)
                location           TEXT,                          -- emplacement (ex: étagère A2)
                tag_number         TEXT,                          -- numéro de tag (RFID/QR...)
                ams_tray           TEXT,                          -- emplacement AMS (slot/cartouche)
                archived           INTEGER NOT NULL DEFAULT 0,    -- 0 actif / 1 archivé
                comment            TEXT,

                -- Champs d'intégration externes
                external_spool_id  TEXT,                          -- identifiant spool côté Spoolman

                FOREIGN KEY (filament_id) REFERENCES filaments(id) ON DELETE CASCADE
            )
            """
        )

        # Ajout rétroactif de colonnes manquantes
        def _maybe_add(table: str, column: str, ddl: str) -> None:
            cur.execute(f"PRAGMA table_info({table})")
            if not any(r[1] == column for r in cur.fetchall()):
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

        _maybe_add("filaments", "profile_id", "TEXT")
        _maybe_add("filaments", "external_filament_id", "TEXT")
        _maybe_add("filaments", "colors_array", "TEXT")
        _maybe_add("filaments", "multicolor_type", "TEXT")

        _maybe_add("bobines", "external_spool_id", "TEXT")

        # Index utiles sur bobines
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bobines_filament ON bobines(filament_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bobines_archived ON bobines(archived)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bobines_extspool ON bobines(external_spool_id)")

        # Trigger de mise à jour du timestamp
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_bobines_updated_at
            AFTER UPDATE ON bobines
            BEGIN
                UPDATE bobines SET updated_at = datetime('now') WHERE id = NEW.id;
            END;
            """
        )
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_bobines_last_used_at
            AFTER UPDATE OF remaining_weight_g ON bobines
            FOR EACH ROW
            WHEN NEW.remaining_weight_g IS NOT OLD.remaining_weight_g
            BEGIN
                UPDATE bobines
                SET last_used_at = datetime('now')
                WHERE id = NEW.id;
            END;
            """
        )

def ensure_filaments_usage_schema() -> None:
    """
    - Ajoute la colonne `ori_spool_id` (TEXT) si absente et la remplit avec spool_id.
    - Crée un index utile sur `ori_spool_id`.
    - (Optionnel) s'assure que spool_id est bien présent (on ne touche pas au type).
    """
    with _tx() as cur:
        # Vérifie les colonnes existantes
        cur.execute("PRAGMA table_info(filament_usage)")
        cols = {row[1] for row in cur.fetchall()}  # {name}

        # Ajoute la colonne si manquante
        if "ori_spool_id" not in cols:
            cur.execute("ALTER TABLE filament_usage ADD COLUMN ori_spool_id TEXT")

            # Initialise ori_spool_id = spool_id pour les lignes existantes
            cur.execute("UPDATE filament_usage SET ori_spool_id = spool_id")

        # Index pratique pour la migration / updates
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_filament_usage_ori_spool_id ON filament_usage(ori_spool_id)"
        )

def migrate_usage_spool_ids_from_external() -> Dict[str, int]:
    """
    Objectif:
      - Conserver l'identifiant externe d'origine dans filaments_usage.ori_spool_id
      - Mettre SYSTÉMATIQUEMENT filaments_usage.spool_id à l'id local de la bobine (bobines.id),
        d'après la correspondance: ori_spool_id == bobines.external_spool_id
        (si pas de correspondance: spool_id = NULL)

    Idempotent:
      - Tu peux relancer la migration sans casser l'état courant.

    Retourne: { 'backfilled_ori', 'updated_spool_id', 'unmatched' }
    """
    ensure_filaments_usage_schema()
    ensure_schema()  # s'assure que bobines/filaments existent

    with _tx() as cur:
        # 1) Backfill ori_spool_id une fois (si encore NULL pour certaines lignes)
        cur.execute("""
            UPDATE filament_usage
               SET ori_spool_id = CAST(spool_id AS TEXT)
             WHERE ori_spool_id IS NULL
               AND spool_id IS NOT NULL
        """)
        backfilled = cur.rowcount or 0

        # 2) Combien de lignes doivent changer ? (comparaison avec la cible)
        #   - cible = b.id si mapping trouvé, sinon NULL
        #   - lignes à mettre à jour = celles où spool_id != cible (NULL compris)
        cur.execute("""
            SELECT COUNT(*)
              FROM filament_usage u
         LEFT JOIN bobines b
                ON CAST(u.ori_spool_id AS TEXT) = CAST(b.external_spool_id AS TEXT)
             WHERE u.ori_spool_id IS NOT NULL
               AND (
                    (b.id IS NULL AND u.spool_id IS NOT NULL)
                 OR (b.id IS NOT NULL AND (u.spool_id IS NULL OR u.spool_id <> b.id))
               )
        """)
        (need_update,) = cur.fetchone()

        # 3) Mise à jour SYSTÉMATIQUE: spool_id := id local mappé, sinon NULL
        cur.execute("""
            UPDATE filament_usage AS u
               SET spool_id = (
                   SELECT b.id
                     FROM bobines b
                    WHERE CAST(u.ori_spool_id AS TEXT) = CAST(b.external_spool_id AS TEXT)
                    LIMIT 1
               )
             WHERE u.ori_spool_id IS NOT NULL
        """)
        # NOTE: rowcount SQLite peut compter large; on renvoie plutôt 'need_update'
        _ = cur.rowcount or 0

        # 4) Combien restent sans correspondance (mapping introuvable) ?
        cur.execute("""
            SELECT COUNT(*)
              FROM filament_usage u
         LEFT JOIN bobines b
                ON CAST(u.ori_spool_id AS TEXT) = CAST(b.external_spool_id AS TEXT)
             WHERE u.ori_spool_id IS NOT NULL
               AND b.id IS NULL
        """)
        (unmatched,) = cur.fetchone()

        return {
            "backfilled_ori": int(backfilled),
            "updated_spool_id": int(need_update),
            "unmatched": int(unmatched),
        }
# ----------------------------------------------------------------------------
# Utilitaires communs
# ----------------------------------------------------------------------------

_FILAMENT_ALLOWED_UPDATE = {
    "name",
    "manufacturer",
    "color",
    "multicolor_type",
    "colors_array",
    "material",
    "price",
    "filament_weight_g",
    "spool_weight_g",
    "comment",
    "external_filament_id",
    "reference_id",
    "profile_id",
    "created_at",
}

_BOBINE_ALLOWED_UPDATE = {
    "filament_id",
    "first_used_at",
    "last_used_at",
    "price_override",
    "remaining_weight_g",
    "location",
    "tag_number",
    "ams_tray",
    "archived",
    "comment",
    "external_spool_id",
    "created_at",
}

def _to_price(value):
    if value is None or value == "":
        return None
    try:
        # accepte string "12.34" ou "12,34"
        s = str(value).replace(",", ".").strip()
        return float(s)
    except Exception:
        return None

def _normalize_hex(h: str | None) -> str | None:
    if not h:
        return None
    h = h.strip().lower()
    if h.startswith("#"):
        h = h[1:]
    if len(h) == 3:
        h = "".join(ch*2 for ch in h)
    h = "".join(c for c in h if c in "0123456789abcdef")
    return h if len(h) == 6 else None

def _normalize_colors_array(colors: list[str] | None):
    """Retourne (color, colors_csv) normalisés (sans '#', triés/dédoublonnés)."""
    if not colors:
        return None, None
    norm = []
    for c in colors:
        nc = _normalize_hex(c)
        if nc:
            norm.append(nc)
    if not norm:
        return None, None
    norm_sorted = sorted(set(norm))  # ordre ignoré pour les doublons
    return norm_sorted[0], ",".join(norm_sorted)

def _filament_duplicate_exists(manufacturer, material, multicolor_type, colors_csv, exclude_id=None) -> bool:
    q = """
        SELECT id
          FROM filaments
         WHERE lower(coalesce(manufacturer,''))   = lower(?)
           AND lower(coalesce(material,''))       = lower(?)
           AND lower(coalesce(multicolor_type,''))= lower(?)
           AND lower(coalesce(colors_array,''))   = lower(?)
    """
    params = [
        (manufacturer or "").strip(),
        (material or "").strip(),
        (multicolor_type or "monochrome").strip(),
        (colors_csv or "").strip(),
    ]
    if exclude_id is not None:
        q += " AND id <> ?"
        params.append(exclude_id)

    with _tx() as cur:
        row = cur.execute(q, params).fetchone()
    return row is not None


def ui_create_filament(payload: dict) -> int:
    """
    Création pour l’UI (ne pas utiliser dans la migration).
    payload: name, manufacturer, material, multicolor_type, colors(list[str]),
             filament_weight_g, spool_weight_g, profile_id, comment, price
    """
    name = (payload.get("name") or "").strip()
    manufacturer = (payload.get("manufacturer") or "").strip()
    material = (payload.get("material") or "").strip()
    multicolor_type = (payload.get("multicolor_type") or "monochrome").strip().lower()
    colors = payload.get("colors") or []

    color, colors_csv = _normalize_colors_array(colors)
    if _filament_duplicate_exists(manufacturer, material, multicolor_type, colors_csv):
        raise ValueError("DUPLICATE_FILAMENT")

    filament_weight_g = int(payload.get("filament_weight_g") or 1000)
    spool_weight_g = int(payload.get("spool_weight_g") or 200)
    profile_id = payload.get("profile_id")
    comment = payload.get("comment")
    price = _to_price(payload.get("price"))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    with _tx() as cur:
        cur.execute("""
            INSERT INTO filaments
            (created_at, updated_at, name, manufacturer, material,
             multicolor_type, color, colors_array,
             filament_weight_g, spool_weight_g, profile_id, comment, price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now, now, name, manufacturer, material,
              multicolor_type, color, colors_csv,
              filament_weight_g, spool_weight_g, profile_id, comment, price))
        return cur.lastrowid


def ui_update_filament(filament_id: int, payload: dict) -> None:
    """
    Édition pour l’UI (ne pas utiliser dans la migration).
    """
    name = (payload.get("name") or "").strip()
    manufacturer = (payload.get("manufacturer") or "").strip()
    material = (payload.get("material") or "").strip()
    multicolor_type = (payload.get("multicolor_type") or "monochrome").strip().lower()
    colors = payload.get("colors") or []

    color, colors_csv = _normalize_colors_array(colors)
    if _filament_duplicate_exists(manufacturer, material, multicolor_type, colors_csv, exclude_id=filament_id):
        raise ValueError("DUPLICATE_FILAMENT")

    filament_weight_g = int(payload.get("filament_weight_g") or 1000)
    spool_weight_g = int(payload.get("spool_weight_g") or 200)
    profile_id = payload.get("profile_id")
    comment = payload.get("comment")
    price = _to_price(payload.get("price"))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    with _tx() as cur:
        cur.execute("""
            UPDATE filaments
               SET updated_at = ?,
                   name = ?, manufacturer = ?, material = ?,
                   multicolor_type = ?, color = ?, colors_array = ?,
                   filament_weight_g = ?, spool_weight_g = ?, profile_id = ?, comment = ?,
                   price = ?
             WHERE id = ?
        """, (now, name, manufacturer, material,
              multicolor_type, color, colors_csv,
              filament_weight_g, spool_weight_g, profile_id, comment,
              price, filament_id))

def _validate_non_negative(name: str, value: Optional[float]) -> None:
    if value is None:
        return
    if value < 0:  # type: ignore[operator]
        raise ValueError(f"{name} ne peut pas être négatif")


# ----------------------------------------------------------------------------
# Filaments – CRUD & helpers
# ----------------------------------------------------------------------------

def add_filament(
    *,
    name: str,
    manufacturer: Optional[str] = None,
    color: Optional[str] = None,
    multicolor_type: Optional[str] = None,
    colors_array: Optional[str] = None,  # CSV string "#RRGGBB,#RRGGBB"
    material: Optional[str] = None,
    price: Optional[float] = None,
    filament_weight_g: Optional[float] = None,
    spool_weight_g: Optional[float] = None,
    comment: Optional[str] = None,
    external_filament_id: Optional[str] = None,
    reference_id: Optional[str] = None,
    profile_id: Optional[str] = None,
    created_at: Optional[str] = None,  # permet d'injecter registered depuis Spoolman
) -> int:
    """Insère un filament et retourne son id."""
    _validate_non_negative("price", price)
    _validate_non_negative("filament_weight_g", filament_weight_g)
    _validate_non_negative("spool_weight_g", spool_weight_g)

    with _tx() as cur:
        if created_at:
            cur.execute(
                """
                INSERT INTO filaments(
                    created_at, name, manufacturer, color, multicolor_type, colors_array, material,
                    price, filament_weight_g, spool_weight_g, comment,
                    external_filament_id, reference_id, profile_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    created_at, name, manufacturer, color, multicolor_type, colors_array, material,
                    price, filament_weight_g, spool_weight_g, comment,
                    external_filament_id, reference_id, profile_id,
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO filaments(
                    name, manufacturer, color, multicolor_type, colors_array, material,
                    price, filament_weight_g, spool_weight_g, comment,
                    external_filament_id, reference_id, profile_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    name, manufacturer, color, multicolor_type, colors_array, material,
                    price, filament_weight_g, spool_weight_g, comment,
                    external_filament_id, reference_id, profile_id,
                ),
            )
        return int(cur.lastrowid)


def update_filament(filament_id: int, **fields: Any) -> None:
    """Met à jour un filament (update partiel)."""
    if not fields:
        return

    invalid = set(fields) - _FILAMENT_ALLOWED_UPDATE
    if invalid:
        raise ValueError(f"Champs non autorisés pour update_filament: {sorted(invalid)}")

    # validations simples
    for k in ("price", "filament_weight_g", "spool_weight_g"):
        if k in fields:
            _validate_non_negative(k, fields[k])

    sets = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values())
    values.append(filament_id)

    with _tx() as cur:
        cur.execute(f"UPDATE filaments SET {sets} WHERE id = ?", values)
        if cur.rowcount == 0:
            raise ValueError(f"Filament introuvable id={filament_id}")


def remove_filament(filament_id: int) -> None:
    """Supprime un filament et ses bobines associées (cascade)."""
    with _tx() as cur:
        cur.execute("DELETE FROM filaments WHERE id = ?", (filament_id,))
        if cur.rowcount == 0:
            raise ValueError(f"Filament introuvable id={filament_id}")


def get_filament(filament_id: int) -> Optional[sqlite3.Row]:
    """Retourne un filament (ou None)."""
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM filaments WHERE id = ?", (filament_id,)).fetchone()
        return row
    finally:
        conn.close()


def list_filaments(
    *,
    manufacturer: Optional[str] = None,
    material: Optional[str] = None,
    search: Optional[str] = None,  # recherchera sur name/manufacturer/material/color
    limit: Optional[int] = None,
    offset: int = 0,
    order_by: str = "created_at DESC",
) -> List[sqlite3.Row]:
    """Liste paginée/filtrée des filaments."""
    where: List[str] = []
    params: List[Any] = []

    if manufacturer:
        where.append("manufacturer = ?")
        params.append(manufacturer)
    if material:
        where.append("material = ?")
        params.append(material)
    if search:
        like = f"%{search}%"
        where.append("(name LIKE ? OR manufacturer LIKE ? OR material LIKE ? OR color LIKE ?)")
        params.extend([like, like, like, like])

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    lim = f" LIMIT {int(limit)}" if limit is not None else ""
    off = f" OFFSET {int(offset)}" if offset and limit is not None else (f" OFFSET {int(offset)}" if offset and limit is None else "")

    sql = f"SELECT * FROM filaments {where_sql} ORDER BY {order_by}{lim}{off}".strip()

    conn = _connect()
    try:
        rows = conn.execute(sql, params).fetchall()
        return list(rows)
    finally:
        conn.close()


def count_filaments() -> int:
    conn = _connect()
    try:
        (n,) = conn.execute("SELECT COUNT(*) FROM filaments").fetchone()
        return int(n)
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# Bobines – CRUD & helpers
# ----------------------------------------------------------------------------

def add_bobine(
    *,
    filament_id: int,
    price_override: Optional[float] = None,
    remaining_weight_g: Optional[float] = None,
    location: Optional[str] = None,
    tag_number: Optional[str] = None,
    ams_tray: Optional[str] = None,
    archived: bool = False,
    comment: Optional[str] = None,
    created_at: Optional[str] = None,  # ISO optionnel si on veut backfiller
    first_used_at: Optional[str] = None,
    last_used_at: Optional[str] = None,
    external_spool_id: Optional[str] = None,
) -> int:
    """Crée une bobine et retourne son id."""
    _validate_non_negative("price_override", price_override)
    _validate_non_negative("remaining_weight_g", remaining_weight_g)

    with _tx() as cur:
        # s'assure que le filament existe
        cur.execute("SELECT 1 FROM filaments WHERE id = ?", (filament_id,))
        if cur.fetchone() is None:
            raise ValueError(f"Filament introuvable id={filament_id}")

        cur.execute(
            """
            INSERT INTO bobines(
                filament_id, price_override, remaining_weight_g, location,
                tag_number, ams_tray, archived, comment,
                created_at, first_used_at, last_used_at, external_spool_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                filament_id, price_override, remaining_weight_g, location,
                tag_number, ams_tray, 1 if archived else 0, comment,
                created_at, first_used_at, last_used_at, external_spool_id,
            ),
        )
        return int(cur.lastrowid)


def update_bobine(bobine_id: int, **fields: Any) -> None:
    """Met à jour une bobine (update partiel)."""
    if not fields:
        return

    invalid = set(fields) - _BOBINE_ALLOWED_UPDATE
    if invalid:
        raise ValueError(f"Champs non autorisés pour update_bobine: {sorted(invalid)}")

    for k in ("price_override", "remaining_weight_g"):
        if k in fields:
            _validate_non_negative(k, fields[k])

    # si on modifie filament_id, vérifier la FK cible
    if "filament_id" in fields and fields["filament_id"] is not None:
        with _tx() as cur:
            cur.execute("SELECT 1 FROM filaments WHERE id = ?", (fields["filament_id"],))
            if cur.fetchone() is None:
                raise ValueError(f"Filament introuvable id={fields['filament_id']}")

    sets = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values())
    values.append(bobine_id)

    with _tx() as cur:
        cur.execute(f"UPDATE bobines SET {sets} WHERE id = ?", values)
        if cur.rowcount == 0:
            raise ValueError(f"Bobine introuvable id={bobine_id}")


def remove_bobine(bobine_id: int) -> None:
    with _tx() as cur:
        cur.execute("DELETE FROM bobines WHERE id = ?", (bobine_id,))
        if cur.rowcount == 0:
            raise ValueError(f"Bobine introuvable id={bobine_id}")


def archive_bobine(bobine_id: int, archived: bool = True) -> None:
    """
    Archive ou désarchive une bobine.

    - Si archived=True :
        - archived = 1
        - location = "Archives"
        - remaining_weight_g = 0
        - ams_tray = ""
    - Si archived=False :
        - archived = 0
        (on ne touche pas aux autres champs)
    """
    if archived:
        update_bobine(
            bobine_id,
            archived=1,
            location="Archives",
            remaining_weight_g=0,
            ams_tray=""
        )
    else:
        update_bobine(
            bobine_id,
            archived=0
        )


def get_bobine(bobine_id: int) -> Optional[sqlite3.Row]:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM bobines WHERE id = ?", (bobine_id,)).fetchone()
        return row
    finally:
        conn.close()


def list_bobines(
    *,
    filament_id: Optional[int] = None,
    archived: Optional[bool] = None,
    ams_tray: Optional[str] = None,
    location: Optional[str] = None,
    search: Optional[str] = None,  # recherche dans commentaire + tag_number
    limit: Optional[int] = None,
    offset: int = 0,
    order_by: str = "created_at DESC",
) -> List[sqlite3.Row]:
    where: List[str] = []
    params: List[Any] = []

    if filament_id is not None:
        where.append("filament_id = ?")
        params.append(filament_id)
    if archived is not None:
        where.append("archived = ?")
        params.append(1 if archived else 0)
    if ams_tray:
        where.append("ams_tray = ?")
        params.append(ams_tray)
    if location:
        where.append("location = ?")
        params.append(location)
    if search:
        like = f"%{search}%"
        where.append("(comment LIKE ? OR tag_number LIKE ?)")
        params.extend([like, like])

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    lim = f" LIMIT {int(limit)}" if limit is not None else ""
    off = f" OFFSET {int(offset)}" if offset and limit is not None else (f" OFFSET {int(offset)}" if offset and limit is None else "")

    sql = f"SELECT * FROM bobines {where_sql} ORDER BY {order_by}{lim}{off}".strip()

    conn = _connect()
    try:
        return list(conn.execute(sql, params).fetchall())
    finally:
        conn.close()


def get_bobine_effective_price(bobine_id: int) -> Optional[float]:
    """Retourne le prix effectif de la bobine (override sinon prix du filament)."""
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT COALESCE(b.price_override, f.price) AS effective_price
            FROM bobines b
            JOIN filaments f ON f.id = b.filament_id
            WHERE b.id = ?
            """,
            (bobine_id,),
        ).fetchone()
        return None if row is None else (None if row[0] is None else float(row[0]))
    finally:
        conn.close()


def set_first_used_if_null(bobine_id: int, when_iso: Optional[str] = None) -> None:
    """Pose `first_used_at` si absent."""
    with _tx() as cur:
        cur.execute(
            """
            UPDATE bobines
               SET first_used_at = COALESCE(first_used_at, COALESCE(?, datetime('now')))
             WHERE id = ?
            """,
            (when_iso, bobine_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Bobine introuvable id={bobine_id}")


def touch_last_used(bobine_id: int, when_iso: Optional[str] = None) -> None:
    """Met `last_used_at` à now (ou valeur fournie)."""
    with _tx() as cur:
        cur.execute(
            "UPDATE bobines SET last_used_at = COALESCE(?, datetime('now')) WHERE id = ?",
            (when_iso, bobine_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Bobine introuvable id={bobine_id}")


def consume_weight(bobine_id: int, grams: float) -> float:
    """Décrémente `remaining_weight_g` (clamp à 0) et retourne le nouveau restant.

    Lève si `grams` est négatif.
    """
    if grams < 0:
        raise ValueError("grams doit être positif")

    with _tx() as cur:
        cur.execute("SELECT remaining_weight_g FROM bobines WHERE id = ?", (bobine_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Bobine introuvable id={bobine_id}")
        current = row[0] if row[0] is not None else 0.0
        new_val = max(0.0, float(current) - float(grams))
        logger.debug('Old weight : ' + str(current) + ' New weight : ' + str(new_val) + ' for spool ' + str(bobine_id))
        cur.execute("UPDATE bobines SET remaining_weight_g = ? WHERE id = ?", (new_val, bobine_id))
        return new_val


def refill_weight(bobine_id: int, grams: float) -> float:
    """Incrémente `remaining_weight_g` (utile pour correction / recharge) et retourne le nouveau restant."""
    if grams < 0:
        raise ValueError("grams doit être positif")
    with _tx() as cur:
        cur.execute("SELECT remaining_weight_g FROM bobines WHERE id = ?", (bobine_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Bobine introuvable id={bobine_id}")
        current = row[0] if row[0] is not None else 0.0
        new_val = float(current) + float(grams)
        cur.execute("UPDATE bobines SET remaining_weight_g = ? WHERE id = ?", (new_val, bobine_id))
        return new_val


def total_remaining_for_filament(filament_id: int, include_archived: bool = False) -> float:
    """Somme des `remaining_weight_g` pour un filament donné."""
    conn = _connect()
    try:
        if include_archived:
            row = conn.execute(
                "SELECT SUM(remaining_weight_g) FROM bobines WHERE filament_id = ?",
                (filament_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT SUM(remaining_weight_g) FROM bobines WHERE filament_id = ? AND archived = 0",
                (filament_id,),
            ).fetchone()
        total = 0.0 if row is None or row[0] is None else float(row[0])
        return total
    finally:
        conn.close()


def count_bobines(*, active_only: bool = False) -> int:
    conn = _connect()
    try:
        if active_only:
            (n,) = conn.execute("SELECT COUNT(*) FROM bobines WHERE archived = 0").fetchone()
        else:
            (n,) = conn.execute("SELECT COUNT(*) FROM bobines").fetchone()
        return int(n)
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# (WIP) Import / Sync Spoolman (placeholder pour future implémentation)
# ----------------------------------------------------------------------------

def sync_from_spoolman(api_url: str, token: Optional[str] = None) -> Dict[str, Any]:
    """Placeholder de sync depuis Spoolman.

    Idée (à implémenter plus tard):
      - GET /api/filaments, /api/spools
      - pour chaque filament/spool, mapper -> add/update locaux
      - conserver le mapping via `external_filament_id` et éventuellement `tag_number`

    Retourne un petit résumé d'opération.
    """
    # volontairement non implémenté ici; on fournira ultérieurement une version
    # utilisant `requests` avec pagination et upserts.
    return {
        "ok": False,
        "message": "sync_from_spoolman non implémenté pour l'instant",
    }


# ----------------------------------------------------------------------------
# Import / Sync Spoolman (v1 REST)
# ----------------------------------------------------------------------------

class _HTTPError(RuntimeError):
    pass


def _http_get(url: str, headers: Optional[Dict[str, str]] = None) -> Any:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise _HTTPError(f"GET {url} failed: {e}")


def _paginate_spoolman(base_url: str, endpoint: str, token: Optional[str]) -> List[Dict[str, Any]]:
    """Récupère toutes les pages d'un endpoint Spoolman.

    Supporte 2 styles courants:
      - DRF-like: {count, next, previous, results}
      - Classic:  {items, total, page, page_size}
    """
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Normalise base_url (pas de slash final)
    base = base_url.rstrip("/")
    url = f"{base}/api/v1/{endpoint.lstrip('/')}"

    out: List[Dict[str, Any]] = []
    seen = set()

    while url:
        data = _http_get(url, headers=headers)
        if isinstance(data, dict):
            if "results" in data:  # DRF-like
                items = data.get("results") or []
                next_url = data.get("next")
            elif "items" in data:  # classic
                items = data.get("items") or []
                # Essaye de construire URL page suivante si dispo
                if data.get("page") and data.get("page_size") and data.get("total"):
                    page = int(data["page"]) + 1
                    page_size = int(data["page_size"])
                    total = int(data["total"])
                    next_url = None
                    if (page - 1) * page_size + len(items) < total:
                        # Reconstruit depuis url courante
                        parsed = urllib.parse.urlparse(url)
                        q = urllib.parse.parse_qs(parsed.query)
                        q["page"] = [str(page)]
                        q["page_size"] = [str(page_size)]
                        next_url = urllib.parse.urlunparse(
                            parsed._replace(query=urllib.parse.urlencode(q, doseq=True))
                        )
                else:
                    next_url = None
            else:
                # Non paginé: tente liste brute
                if isinstance(data, list):
                    items = data
                else:
                    items = [data]
                next_url = None
        elif isinstance(data, list):
            items = data
            next_url = None
        else:
            items = []
            next_url = None

        for it in items:
            # évite doublons basés sur (endpoint,id)
            rid = (endpoint, json.dumps(it.get("id", it), sort_keys=True))
            if rid not in seen:
                out.append(it)
                seen.add(rid)
        url = next_url

    return out


# ------------------------------ Mapping helpers ------------------------------

def _spoolman_vendor_name(f: Dict[str, Any]) -> Optional[str]:
    v = f.get("vendor")
    if isinstance(v, dict):
        return v.get("name")
    return None

    if isinstance(v, dict):
        return v.get("name") or v.get("display_name")
    return f.get("vendor_name")


def _spoolman_material(f: Dict[str, Any]) -> Optional[str]:
    m = f.get("material")
    if isinstance(m, dict):
        return m.get("name") or m.get("type")
    return f.get("material")


def _spoolman_color(f: Dict[str, Any]) -> Optional[str]:
    return f.get("color_hex")



def _spoolman_price(f: Dict[str, Any]) -> Optional[float]:
    return f.get("price") or f.get("price_eur") or f.get("msrp")


def _spoolman_filament_weight(f: Dict[str, Any]) -> Optional[float]:
    return (
        f.get("weight_g")
        or f.get("filament_weight_g")
        or f.get("net_weight_g")
        or f.get("weight")
    )


def _spoolman_spool_weight(f: Dict[str, Any]) -> Optional[float]:
    return f.get("spool_weight_g") or f.get("spool_weight")


# ------------------------------ Upserts ------------------------------

def _clean_profile_id(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s or None


def _clean_quoted(val: Any) -> Optional[str]:
    """Nettoie les strings de Spoolman parfois encadrées de quotes doubles dans JSON."""
    if val is None:
        return None
    s = str(val)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s or None


def _upsert_filament_from_spoolman_exact(f: Dict[str, Any]) -> int:
    """Crée/Met à jour un filament local depuis l'objet Spoolman selon le mapping imposé.

    Mapping:
      - created_at           <- registered
      - name                 <- name
      - manufacturer         <- vendor.name
      - color                <- color_hex
      - material             <- material
      - price                <- price
      - spool_weight_g       <- spool_weight
      - filament_weight_g    <- weight
      - profile_id           <- extra.filament_id (nettoyé)
      - colors_array         <- multi_color_hexes (CSV, inchangé)
      - multicolor_type      <- multi_color_direction (ou 'none')
      - comment              <- append "Importé depuis Spoolman le <now>"
      - external_filament_id <- id (string)
    """
    external_id = str(f.get("id")) if f.get("id") is not None else None

    # multi-couleur
    colors_array = f.get("multi_color_hexes")
    multicolor_type = f.get("multi_color_direction") or ("none" if colors_array in (None, "") else None)

    # profile id dans extra
    extra = f.get("extra") or {}
    profile_id = _clean_profile_id(extra.get("filament_id"))
    payload = {
        "created_at": f.get("registered"),
        "name": f.get("name"),
        "manufacturer": _spoolman_vendor_name(f),
        "color": _spoolman_color(f),
        "material": _spoolman_material(f),
        "price": _spoolman_price(f),
        "spool_weight_g": _spoolman_spool_weight(f),
        "filament_weight_g": _spoolman_filament_weight(f),
        "profile_id": profile_id,
        "colors_array": f.get("multi_color_hexes"),
        "multicolor_type": multicolor_type,
        "comment": f"Importé depuis Spoolman le {datetime.now().isoformat(timespec='seconds')}",
        "external_filament_id": external_id,
    }
    # Cherche un filament existant par external_filament_id
    conn = _connect()
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT id, comment FROM filaments WHERE external_filament_id = ?",
            (external_id,),
        ).fetchone() if external_id else None
    finally:
        conn.close()

    if row:
        # On append le tag d'import au commentaire existant
        fields = {k: v for k, v in payload.items() if k in _FILAMENT_ALLOWED_UPDATE and k != "comment"}
        update_filament(int(row[0]), **fields)
        return int(row[0])
    else:
        # Insertion avec created_at forcé depuis registered
        return add_filament(**payload)

def _clean_profile_id(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val)
    # Certains payloads contiennent des quotes incluses, ex: '"GFA00"' -> 'GFA00'
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s or None


def _upsert_filament_from_spoolman(f: Dict[str, Any]) -> int:
    """Crée/Met à jour un filament local depuis l'objet Spoolman et retourne son id local."""
    external_id = str(f.get("id")) if f.get("id") is not None else None
    name = f.get("name") or f.get("display_name") or "Sans nom"

    payload = {
        "name": name,
        "manufacturer": _spoolman_vendor_name(f),
        "color": _spoolman_color(f),
        "material": _spoolman_material(f),
        "price": _spoolman_price(f),
        "filament_weight_g": _spoolman_filament_weight(f),
        "spool_weight_g": _spoolman_spool_weight(f),

        "external_filament_id": external_id,
        "reference_id": f.get("sku") or f.get("reference") or None,
    }

    # Cherche un filament existant par external_filament_id
    conn = _connect()
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT id FROM filaments WHERE external_filament_id = ?",
            (external_id,),
        ).fetchone() if external_id else None
    finally:
        conn.close()

    if row:
        update_filament(int(row[0]), **{k: v for k, v in payload.items() if k in _FILAMENT_ALLOWED_UPDATE})
        return int(row[0])
    else:
        return add_filament(**payload)


def _upsert_bobine_from_spoolman_exact(s: Dict[str, Any], filament_local_id: int) -> int:
    """Crée/Met à jour une bobine locale depuis l'objet *spool* Spoolman selon le mapping suivant:

    Mapping bobine:
      - created_at         <- registered
      - first_used_at      <- first_used
      - last_used_at       <- last_used
      - price_override     <- price
      - remaining_weight_g <- remaining_weight
      - location           <- location
      - tag_number         <- extra.tag (nettoyé via _clean_quoted)
      - ams_tray           <- extra.active_tray (nettoyé)
      - archived           <- archived (bool)
      - comment            <- append "Importé depuis Spoolman le <now>"
      - external_spool_id  <- id (string)
      - filament_id        <- `filament_local_id` (résolu en amont)
    """
    external_spool_id = str(s.get("id")) if s.get("id") is not None else None
    extra = s.get("extra") or {}

    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    import_tag = f"Importé depuis Spoolman le {now_iso}"

    payload = {
        "filament_id": filament_local_id,
        "price_override": s.get("price"),
        "remaining_weight_g": s.get("remaining_weight"),
        "location": s.get("location"),
        "tag_number": _clean_quoted(extra.get("tag")),
        "ams_tray": _clean_quoted(extra.get("active_tray")),
        "archived": bool(s.get("archived", False)),
        "created_at": s.get("registered"),
        "first_used_at": s.get("first_used"),
        "last_used_at": s.get("last_used"),
        "external_spool_id": external_spool_id,
    }

    # Cherche par external_spool_id
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, comment FROM bobines WHERE external_spool_id = ?",
            (external_spool_id,),
        ).fetchone() if external_spool_id else None
    finally:
        conn.close()

    if row:
        fields = {
            k: v for k, v in payload.items()
            if k in _BOBINE_ALLOWED_UPDATE and k not in {"created_at", "comment"}
        }
        update_bobine(int(row[0]), **fields)
        return int(row[0])
    else:
        return add_bobine(**payload)


def sync_from_spoolman(base_url: str, token: Optional[str] = None) -> Dict[str, Any]:
    """Synchronise depuis Spoolman -> base locale.

    Paramètres:
      - base_url: URL de base de Spoolman (ex: http://host:port)
      - token   : Bearer token si l'instance en requiert un

    Comportement:
      - Récupère tous les *filaments* puis toutes les *spools* (bobines) via `/api/v1/`.
      - Upsert côté local via `external_filament_id` et `external_spool_id`.
      - Fait au mieux quel que soit le schéma exact (clés déduites de façon souple).

    Retourne un résumé de l'opération.
    """
    ensure_schema()

    filaments_remote = _paginate_spoolman(base_url, "filament", token)
    spools_remote    = _paginate_spoolman(base_url, "spool?allow_archived=1", token)

    filament_map: Dict[str, int] = {}
    created_f = updated_f = 0

    # Upsert des filaments
    for f in filaments_remote:
        # existence approximative, si pas d'external id on s'en remettra au nom
        had_local = False
        if f.get("id") is not None:
            conn = _connect()
            try:
                r = conn.execute("SELECT 1 FROM filaments WHERE external_filament_id = ?", (str(f["id"]),)).fetchone()
                had_local = r is not None
            finally:
                conn.close()
        local_id = _upsert_filament_from_spoolman_exact(f)
        filament_map[str(f.get("id"))] = local_id
        if had_local:
            updated_f += 1
        else:
            created_f += 1

    # Upsert des spools/bobines
    created_s = updated_s = 0
    for s in spools_remote:
        # map filament_id distant -> local
        remote_fid = s.get("filament_id") or (s.get("filament") and s["filament"].get("id"))
        if remote_fid is None:
            # impossible d'ancrer la bobine -> on ignore
            continue
        local_fid = filament_map.get(str(remote_fid))
        if not local_fid:
            # si filament non sync (cas limite), on tente un upsert direct depuis nested obj
            if isinstance(s.get("filament"), dict):
                local_fid = _upsert_filament_from_spoolman_exact(s["filament"])  # type: ignore[arg-type]
            else:
                # sécurité: skip
                continue

        # Détermine existence locale
        before = False
        if s.get("id") is not None:
            conn = _connect()
            try:
                r = conn.execute(
                    "SELECT 1 FROM bobines WHERE external_spool_id = ?",
                    (str(s["id"]),),
                ).fetchone()
                before = r is not None
            finally:
                conn.close()

        _upsert_bobine_from_spoolman_exact(s, local_fid)
        if before:
            updated_s += 1
        else:
            created_s += 1

    summary = {
        "filaments": {"created": created_f, "updated": updated_f, "total": len(filaments_remote)},
        "spools": {"created": created_s, "updated": updated_s, "total": len(spools_remote)},
    }

    # Migration des usages sur IDs locaux
    usage_mig = migrate_usage_spool_ids_from_external()
    summary["filaments_usage_migration"] = usage_mig
    return summary
    
def update_field_spool(
    *,
    field: str,
    value: Any,
    bobine_id: Optional[int] = None,
    external_spool_id: Optional[str] = None,
) -> int:
    """
    Met à jour UN champ d'une bobine (spool) identifiée soit par id local (bobine_id),
    soit par id externe Spoolman (external_spool_id). Retourne l'id local de la bobine.

    - Le champ doit appartenir à _BOBINE_ALLOWED_UPDATE.
    - Validations incluses :
        * price_override, remaining_weight_g : non négatifs
        * filament_id : FK existante si non-NULL
        * archived : normalisé en 0/1 (accepte bool/int/str)
    - Ne modifie rien d'autre (pas de side-effects).

    Exceptions :
        - ValueError si identifiant manquant/ambigu, si bobine introuvable,
          si champ non autorisé, ou si validations échouent.
    """
    # 1) identification : exactement un identifiant
    if (bobine_id is None) == (external_spool_id is None):
        raise ValueError("Fournir exactement un identifiant: bobine_id OU external_spool_id")

    # 2) validation du champ
    if field not in _BOBINE_ALLOWED_UPDATE:
        raise ValueError(f"Champ non autorisé pour bobine: {field}")

    # 3) résolution de l'id local
    local_id: Optional[int] = None
    if bobine_id is not None:
        local_id = int(bobine_id)
        row = get_bobine(local_id)
        if row is None:
            raise ValueError(f"Bobine introuvable id={local_id}")
    else:
        conn = _connect()
        try:
            r = conn.execute(
                "SELECT id FROM bobines WHERE external_spool_id = ?",
                (str(external_spool_id),),
            ).fetchone()
            if r is None:
                raise ValueError(f"Bobine introuvable external_spool_id={external_spool_id}")
            local_id = int(r[0])
        finally:
            conn.close()

    assert local_id is not None  # pour mypy

    # 4) validations métier spécifiques
    if field in ("price_override", "remaining_weight_g"):
        _validate_non_negative(field, value)

    if field == "filament_id" and value is not None:
        # vérifie la FK vers filaments
        conn = _connect()
        try:
            chk = conn.execute("SELECT 1 FROM filaments WHERE id = ?", (int(value),)).fetchone()
            if chk is None:
                raise ValueError(f"Filament introuvable id={value}")
        finally:
            conn.close()

    if field == "archived":
        # normaliser vers 0/1
        if isinstance(value, bool):
            value = 1 if value else 0
        elif isinstance(value, (int, float)):
            value = 1 if int(value) != 0 else 0
        elif isinstance(value, str):
            v = value.strip().lower()
            value = 1 if v in ("1", "true", "yes", "y", "on") else 0
        else:
            # par défaut, falsy -> 0, truthy -> 1
            value = 1 if value else 0

    # 5) update via la fonction existante (respecte la transaction & les triggers)
    update_bobine(local_id, **{field: value})
    return local_id

def _split_colors_array(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    # accepte CSV "0047BB,BB22A3" (avec ou sans #) et nettoie les espaces
    arr = [p.strip() for p in str(s).split(",")]
    arr = [p for p in arr if p]
    return arr or None

def fetch_spools(*, archived: bool = False) -> List[Dict[str, Any]]:
    """
    Retourne la liste des bobines (spools) locales, chacune enrichie avec son filament,
    et les champs dérivés attendus par l'ancien code fetchSpools().

    Paramètres:
      - archived:
          False (par défaut) -> bobines actives uniquement
          True               -> toutes les bobines (actives + archivées)

    Sortie: liste de dicts JSON-compatibles, chaque spool contient aussi son `filament`.
    """
    base_sql = """
      SELECT
        b.id              AS b_id,
        b.filament_id     AS b_filament_id,
        b.created_at      AS b_created_at,
        b.first_used_at   AS b_first_used_at,
        b.last_used_at    AS b_last_used_at,
        b.price_override  AS b_price_override,
        b.remaining_weight_g AS b_remaining_weight_g,
        b.location        AS b_location,
        b.tag_number      AS b_tag_number,
        b.ams_tray        AS b_ams_tray,
        b.archived        AS b_archived,
        b.comment         AS b_comment,
        b.external_spool_id AS b_external_spool_id,

        f.id              AS f_id,
        f.name            AS f_name,
        f.manufacturer    AS f_manufacturer,
        f.profile_id        AS f_profile_id,
        f.material        AS f_material,
        f.color           AS f_color,
        f.price           AS f_price,
        f.filament_weight_g AS f_filament_weight_g,
        f.spool_weight_g  AS f_spool_weight_g,
        f.colors_array    AS f_colors_array,
        f.multicolor_type AS f_multicolor_type
      FROM bobines b
      JOIN filaments f ON f.id = b.filament_id
    """

    if archived:
        sql = base_sql  # toutes les bobines
        params = ()
    else:
        sql = base_sql + " WHERE b.archived = 0"  # actives uniquement
        params = ()

    sql += " ORDER BY b.created_at DESC, b.id DESC"

    conn = _connect()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    out: List[Dict[str, Any]] = []
    for r in rows:
        filament_weight = float(r["f_filament_weight_g"]) if r["f_filament_weight_g"] is not None else 0.0
        initial_weight = filament_weight

        price_override = r["b_price_override"]
        filament_price = r["f_price"] if r["f_price"] is not None else 20.0
        price = float(price_override) if price_override is not None else float(filament_price)
        if price is None:
            price = 20.0

        cost_per_gram = (price / initial_weight) if (initial_weight and price >= 0) else 0.02
        filament_cost_per_gram = (
            (filament_price / filament_weight) if (filament_weight and filament_price >= 0) else 0.02
        )

        multi_list = _split_colors_array(r["f_colors_array"])

        spool: Dict[str, Any] = {
            "id": int(r["b_id"]),
            "external_spool_id": r["b_external_spool_id"],
            "filament_id": int(r["b_filament_id"]),
            "created_at": r["b_created_at"],
            "first_used": r["b_first_used_at"],
            "last_used": r["b_last_used_at"],
            "remaining_weight": (float(r["b_remaining_weight_g"]) if r["b_remaining_weight_g"] is not None else None),
            "location": r["b_location"],
            "tag_number": r["b_tag_number"],
            "ams_tray": r["b_ams_tray"],
            "archived": bool(r["b_archived"]),
            "comment": r["b_comment"],

            "initial_weight": float(initial_weight) if initial_weight else 0.0,
            "price": float(price),
            "filament_price": float(filament_price),
            "cost_per_gram": float(cost_per_gram),
            "filament_cost_per_gram": float(filament_cost_per_gram),
            "extra" :{
                "tag":r["b_tag_number"],
                "active_tray":r["b_ams_tray"]
            },
            "filament": {
                "id": int(r["f_id"]),
                "name": r["f_name"],
                "manufacturer": r["f_manufacturer"],
                "material": r["f_material"],
                "profile_id": r["f_profile_id"],
                "color_hex": r["f_color"],
                "weight": (float(r["f_filament_weight_g"]) if r["f_filament_weight_g"] is not None else None),
                "spool_weight": (float(r["f_spool_weight_g"]) if r["f_spool_weight_g"] is not None else None),
                "price": (float(r["f_price"]) if r["f_price"] is not None else None),
                "multi_color_hexes": multi_list,
                "multi_color_direction": r["f_multicolor_type"],
                "vendor": {
                    "name": r["f_manufacturer"]  # même si None, l'objet existe → pas d’UndefinedError
                },
                "extra" :{
                    "filament_id":r["f_profile_id"]
                }
            }
        }
        out.append(spool)

    return out

from typing import Any, Dict, Optional

def fetch_spool_by_id(spool_id: int) -> Optional[Dict[str, Any]]:
    """
    Retourne une seule bobine locale avec son filament enrichi,
    au format identique à fetch_spools().
    Si l'id n'existe pas → None.
    """
    sql = """
      SELECT
        b.id              AS b_id,
        b.filament_id     AS b_filament_id,
        b.created_at      AS b_created_at,
        b.first_used_at   AS b_first_used_at,
        b.last_used_at    AS b_last_used_at,
        b.price_override  AS b_price_override,
        b.remaining_weight_g AS b_remaining_weight_g,
        b.location        AS b_location,
        b.tag_number      AS b_tag_number,
        b.ams_tray        AS b_ams_tray,
        b.archived        AS b_archived,
        b.comment         AS b_comment,
        b.external_spool_id AS b_external_spool_id,

        f.id              AS f_id,
        f.name            AS f_name,
        f.manufacturer    AS f_manufacturer,
        f.profile_id      AS f_profile_id,
        f.material        AS f_material,
        f.color           AS f_color,
        f.price           AS f_price,
        f.filament_weight_g AS f_filament_weight_g,
        f.spool_weight_g  AS f_spool_weight_g,
        f.colors_array    AS f_colors_array,
        f.multicolor_type AS f_multicolor_type
      FROM bobines b
      JOIN filaments f ON f.id = b.filament_id
      WHERE b.id = ?
    """

    conn = _connect()
    try:
        row = conn.execute(sql, (spool_id,)).fetchone()
    finally:
        conn.close()

    if not row:
        return None

    filament_weight = float(row["f_filament_weight_g"]) if row["f_filament_weight_g"] is not None else 0.0
    initial_weight = filament_weight

    price_override = row["b_price_override"]
    filament_price = row["f_price"] if row["f_price"] is not None else 20.0
    price = float(price_override) if price_override is not None else float(filament_price)
    if price is None:
        price = 20.0

    cost_per_gram = (price / initial_weight) if (initial_weight and price >= 0) else 0.02
    filament_cost_per_gram = (
        (filament_price / filament_weight) if (filament_weight and filament_price >= 0) else 0.02
    )

    multi_list = _split_colors_array(row["f_colors_array"])

    spool: Dict[str, Any] = {
        "id": int(row["b_id"]),
        "external_spool_id": row["b_external_spool_id"],
        "filament_id": int(row["b_filament_id"]),
        "created_at": row["b_created_at"],
        "first_used": row["b_first_used_at"],
        "last_used": row["b_last_used_at"],
        "remaining_weight": (float(row["b_remaining_weight_g"]) if row["b_remaining_weight_g"] is not None else None),
        "location": row["b_location"],
        "tag_number": row["b_tag_number"],
        "ams_tray": row["b_ams_tray"],
        "archived": bool(row["b_archived"]),
        "comment": row["b_comment"],

        "initial_weight": float(initial_weight) if initial_weight else 0.0,
        "price": float(price),
        "filament_price": float(filament_price),
        "cost_per_gram": float(cost_per_gram),
        "filament_cost_per_gram": float(filament_cost_per_gram),
        "extra": {
            "tag": row["b_tag_number"],
            "active_tray": row["b_ams_tray"]
        },
        "filament": {
            "id": int(row["f_id"]),
            "name": row["f_name"],
            "manufacturer": row["f_manufacturer"],
            "material": row["f_material"],
            "profile_id": row["f_profile_id"],
            "color_hex": row["f_color"],
            "weight": (float(row["f_filament_weight_g"]) if row["f_filament_weight_g"] is not None else None),
            "spool_weight": (float(row["f_spool_weight_g"]) if row["f_spool_weight_g"] is not None else None),
            "price": (float(row["f_price"]) if row["f_price"] is not None else None),
            "multi_color_hexes": multi_list,
            "multi_color_direction": row["f_multicolor_type"],
            "vendor": {
                "name": row["f_manufacturer"]
            },
            "extra": {
                "filament_id": row["f_profile_id"]
            }
        }
    }

    return spool


def patchLocation(spool_id, ams_id='', tray_id=''):
    location = ''
    ams_name = 'AMS_' + str(ams_id)
    mapping = get_app_setting("LOCATION_MAPPING", "")
    if mapping:
        d = dict(item.split(":", 1) for item in mapping.split(";"))
        if ams_name in d:
            location = d[ams_name] if ams_id == 100 else d[ams_name] + ' ' + str(tray_id)
    update_field_spool(field="location",value=location,bobine_id=spool_id)
    return

def setActiveTray(spool_id, ams_id, tray_id):
    bobine = get_bobine(spool_id)
    
    if bobine["ams_tray"] != trayUid(ams_id, tray_id):
        update_field_spool(field="ams_tray",value=trayUid(ams_id, tray_id),bobine_id=spool_id)
    
    if (int(tray_id) >200):
        patchLocation(spool_id,ams_id)
    else:
        patchLocation(spool_id,ams_id,int(tray_id)+1)
    
    # Remove active tray from inactive spools
    for old_spool in fetch_spools():
        if spool_id != old_spool["id"] and old_spool["ams_tray"] == trayUid(ams_id, tray_id):
            update_field_spool(field="ams_tray",value="",bobine_id=old_spool["id"])
            patchLocation(old_spool["id"],100)
        
def clearActiveTray(ams_id,tray_id):
    for old_spool in fetch_spools():
      if old_spool["ams_tray"] == trayUid(ams_id, tray_id):
        update_field_spool(field="ams_tray",value="",bobine_id=old_spool["id"])
        patchLocation(old_spool["id"],100)

def augmentTrayData(spool_list, tray_data, tray_id):
    tray_data["matched"] = False
    for spool in spool_list:
        if spool.get("extra") and spool["extra"].get("active_tray") and spool["extra"]["active_tray"] == tray_id:
            #TODO: check for mismatch
            tray_data["name"] = spool["filament"]["name"]
            tray_data["vendor"] = spool["filament"]["vendor"]["name"]
            tray_data["remaining_weight"] = spool["remaining_weight"]
        
            if "last_used" in spool:
                raw = spool["last_used"]
                dt = None
            
                # Liste des formats possibles, du plus spécifique au plus générique
                formats = [
                    "%Y-%m-%dT%H:%M:%S.%fZ",   # Avec millisecondes
                    "%Y-%m-%dT%H:%M:%SZ",      # Sans millisecondes
                    "%Y-%m-%d %H:%M:%S",       # Format sans T ni Z
                ]
            
                for fmt in formats:
                    try:
                        dt = datetime.strptime(raw, fmt)
                        break
                    except ValueError:
                        continue

                if dt is None:
                    raise ValueError(f"Format de date non reconnu : {raw}")
            
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                local_time = dt.astimezone()
                tray_data["last_used"] = local_time.strftime("%d.%m.%Y %H:%M:%S")
        
            else:
                tray_data["last_used"] = "-"
            filament = spool.get("filament") or {}

            multi_hexes = filament.get("multi_color_hexes")
            multi_dir = filament.get("multi_color_direction")
            if multi_hexes and multi_dir and str(multi_dir).lower() != "none":
                tray_data["tray_color"] = spool["filament"]["multi_color_hexes"]
                tray_data["tray_color_orientation"] = spool["filament"]["multi_color_direction"]
            else:
                tray_data["tray_color"] = spool["filament"]["color_hex"]
                
            tray_data["matched"] = True
            break
    
    if tray_data.get("tray_type") and tray_data["tray_type"] != "" and tray_data["matched"] == False:
        tray_data["issue"] = True
    else:
        tray_data["issue"] = False

def getAMSFromTray(n):
    return n // 4
    
def spendFilaments(printdata):
    if printdata["ams_mapping"]:
        ams_mapping = printdata["ams_mapping"]
    else:
        ams_mapping = [EXTERNAL_SPOOL_ID]
    
    """
    "ams_mapping": [
                1,
                0,
                -1,
                -1,
                -1,
                1,
                0
            ],
    """
    tray_id = EXTERNAL_SPOOL_ID
    ams_id = EXTERNAL_SPOOL_AMS_ID
    
    ams_usage = []
    filamentOrder = printdata["filamentOrder"]
    #filament_id_to_amstray = {fid: tray for tray, fid in filamentOrder.items()}
    cleaned_mapping = [x for x in ams_mapping if x != -1]
    for filamentId, filament in printdata["filaments"].items():
        if ams_mapping[0] != EXTERNAL_SPOOL_ID:
            try:
                ams_mapping_idx = filamentId - 1
                tray_id = cleaned_mapping[ams_mapping_idx]   # get tray_id from ams_mapping for filament
                ams_id = getAMSFromTray(tray_id)        # caclulate ams_id from tray_id
                tray_id = tray_id - ams_id * 4          # correct tray_id for ams
            except Exception as e:
                continue #filament not used
        
        #if ams_usage.get(trayUid(ams_id, tray_id)):
        #    ams_usage[trayUid(ams_id, tray_id)]["usedGrams"] += float(filament["used_g"])
        #else:
        ams_usage.append({"trayUid": trayUid(ams_id, tray_id), "id": filamentId, "usedGrams":float(filament["used_g"])})
    
    for spool in fetch_spools():
        #TODO: What if there is a mismatch between AMS and SpoolMan?
                    
        if spool.get("extra") and spool.get("extra").get("active_tray"):
            #filament = ams_usage.get()
            active_tray = spool.get("extra").get("active_tray")
            logger.debug('Searching usage for ' + str(spool))
            # iterate over all ams_trays and set spool in print history, at the same time sum the usage for the tray and consume it from the spool
            used_grams = 0
            #print(ams_usage)
            for ams_tray in ams_usage:
                if active_tray == ams_tray["trayUid"]:
                    used_grams += ams_tray["usedGrams"]
                    logger.debug('Found usage for ' + active_tray)
                    update_filament_spool(printdata["print_id"], ams_tray["id"], spool["id"])
            logger.debug('Used Grams ' + str(used_grams))
                
            if used_grams != 0:
                consume_weight(spool["id"], used_grams)

ensure_schema()
