import sqlite3
from typing import Optional, Dict, Any, List, Tuple, NamedTuple, Literal
from datetime import datetime, timezone

# On réutilise la config DB telle qu'elle existe déjà dans le projet
from print_history import db_config

# ---------------------------------------------------------------------------
# Connexion
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    # Active la gestion des clés étrangères (nécessaire pour ON DELETE CASCADE)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---------------------------------------------------------------------------
# Schéma
# ---------------------------------------------------------------------------

def ensure_schema() -> None:
    """
    Crée les tables objects et tag_objects si elles n'existent pas.
    - tag_objects est strictement équivalente à print_tags mais pour les objects.
    """
    conn = _connect()
    cur = conn.cursor()

    # Table des objets (unitaires)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS objects (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_type      TEXT    NOT NULL CHECK (parent_type IN ('print','group')),
            parent_id        INTEGER NOT NULL,
            name             TEXT    NOT NULL,
            cost_fabrication REAL    NOT NULL DEFAULT 0,   -- coût de fabrication (main d'œuvre/élec agrégé selon ton calcul)
            cost_accessory   REAL    NOT NULL DEFAULT 0,   -- coût des accessoires
            cost_total       REAL    NOT NULL DEFAULT 0,   -- coût total = fabrication + accessoires
            available        INTEGER NOT NULL DEFAULT 1,   -- 1 = disponible / 0 = retiré du stock (ex: vendu)
            sold_price       REAL,                         -- prix de vente (si vendu)
            sold_date        TEXT,                         -- date/heure de vente (ISO, datetime('now') si besoin)
            comment          TEXT,                         -- commentaire libre
            created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at       TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    cur.execute("PRAGMA table_info(objects)")
    cols = {r[1] for r in cur.fetchall()}
    if "thumbnail" not in cols:
        cur.execute("ALTER TABLE objects ADD COLUMN thumbnail TEXT")

    # Index utiles
    cur.execute("""CREATE INDEX IF NOT EXISTS idx_objects_parent
                   ON objects(parent_type, parent_id)""")
    cur.execute("""CREATE INDEX IF NOT EXISTS idx_objects_available
                   ON objects(available)""")
    cur.execute("""CREATE INDEX IF NOT EXISTS idx_objects_name
                   ON objects(name)""")

    # Table des tags d'objets (strictement comme print_tags)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tag_objects (
            object_id INTEGER NOT NULL,
            tag       TEXT    NOT NULL,
            PRIMARY KEY (object_id, tag),
            FOREIGN KEY (object_id) REFERENCES objects(id) ON DELETE CASCADE
        )
    """)

    cur.execute("""CREATE INDEX IF NOT EXISTS idx_tag_objects_object
                   ON tag_objects(object_id)""")
    cur.execute("""CREATE INDEX IF NOT EXISTS idx_tag_objects_tag
                   ON tag_objects(tag)""")

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Helpers : lecture / écriture simples
# ---------------------------------------------------------------------------

SourceType = Literal["print", "group"]

class SourceSnapshot(NamedTuple):
    name: str
    number_of_items: int
    full_cost_unit: float
    thumbnail: str | None
    tags: list[str]

def _compute_full_unit(units: int, full_cost_by_item, full_cost) -> float:
    try:
        if full_cost_by_item is not None:
            return float(full_cost_by_item)
        if full_cost is not None and units:
            return float(full_cost) / max(1, int(units))
    except Exception:
        pass
    return 0.0

def _snapshot_from_print(print_id: int) -> SourceSnapshot:
    conn = _connect(); cur = conn.cursor()
    # Lis ce dont on a besoin directement dans prints
    cur.execute("""
        SELECT
            id,
            COALESCE(file_name, 'Print #' || id) AS name,
            COALESCE(number_of_items, 1)        AS number_of_items,
            full_cost_by_item,
            full_cost,
            image_file
        FROM prints
        WHERE id = ?
    """, (print_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return SourceSnapshot(f"Print #{print_id}", 1, 0.0, None, [])

    units = int(row["number_of_items"] or 1)
    full_unit = _compute_full_unit(units, row["full_cost_by_item"], row["full_cost"])
    # On stocke une URL utilisable directement ; adapte si tu préfères stocker juste le filename
    thumb = f"/static/prints/{row['image_file']}" if row["image_file"] else None

    # Tags du print
    cur.execute("SELECT tag FROM print_tags WHERE print_id = ? ORDER BY LOWER(tag)", (print_id,))
    tags = [r[0] for r in cur.fetchall()]
    conn.close()

    return SourceSnapshot(row["name"], max(1, units), full_unit, thumb, tags)

def _snapshot_from_group(group_id: int) -> SourceSnapshot:
    conn = _connect(); cur = conn.cursor()
    # Lis ce dont on a besoin dans print_groups
    cur.execute("""
        SELECT
            id,
            COALESCE(name, 'Groupe ' || id) AS name,
            COALESCE(number_of_items, 1)    AS number_of_items,
            full_cost_by_item,
            full_cost,
            primary_print_id
        FROM print_groups
        WHERE id = ?
    """, (group_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return SourceSnapshot(f"Groupe #{group_id}", 1, 0.0, None, [])

    units = int(row["number_of_items"] or 1)
    full_unit = _compute_full_unit(units, row["full_cost_by_item"], row["full_cost"])

    # Thumbnail du groupe = image du print “référence” s’il existe
    thumb = None
    if row["primary_print_id"]:
        cur.execute("SELECT image_file FROM prints WHERE id = ?", (row["primary_print_id"],))
        p = cur.fetchone()
        if p and p["image_file"]:
            thumb = f"/static/prints/{p['image_file']}"

    # Tags = union des tags de tous les prints du groupe
    cur.execute("""
        SELECT DISTINCT pt.tag
        FROM print_tags pt
        JOIN prints p ON p.id = pt.print_id
        WHERE p.group_id = ?
        ORDER BY LOWER(pt.tag)
    """, (group_id,))
    tags = [r[0] for r in cur.fetchall()]

    conn.close()
    return SourceSnapshot(row["name"], max(1, units), full_unit, thumb, tags)

def snapshot_source(source_type: SourceType, source_id: int) -> SourceSnapshot:
    if source_type == "print":
        return _snapshot_from_print(source_id)
    elif source_type == "group":
        return _snapshot_from_group(source_id)
    else:
        # Type inconnu => snapshot vide
        return SourceSnapshot(f"{source_type} #{source_id}", 1, 0.0, None, [])
    
def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(row) if row else {}


def create_object(
    parent_type: str,
    parent_id: int,
    name: str,
    cost_fabrication: float = 0.0,
    cost_accessory: float = 0.0,
    available: bool = True,
    sold_price: Optional[float] = None,
    sold_date: Optional[str] = None,
    comment: Optional[str] = None,
) -> int:
    """
    Crée un objet unitaire. cost_total est calculé = fabrication + accessoires.
    Retourne l'id de l'objet.
    """
    if parent_type not in ("print", "group"):
        raise ValueError("parent_type doit être 'print' ou 'group'")

    cost_total = float(cost_fabrication or 0) + float(cost_accessory or 0)

    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO objects (
            parent_type, parent_id, name,
            cost_fabrication, cost_accessory, cost_total,
            available, sold_price, sold_date, comment
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            parent_type, int(parent_id), name.strip(),
            float(cost_fabrication or 0), float(cost_accessory or 0), cost_total,
            1 if available else 0,
            float(sold_price) if sold_price is not None else None,
            sold_date,  # laisser None ou une chaîne ISO (ex: datetime('now') côté SQL plus tard si tu préfères)
            comment.strip() if comment else None,
        ),
    )
    obj_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(obj_id)


def get_object(object_id: int) -> Dict[str, Any]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM objects WHERE id = ?", (int(object_id),))
    row = cur.fetchone()
    conn.close()
    return _row_to_dict(row)


def update_object_fields(object_id: int, **fields) -> None:
    """
    Met à jour des champs arbitraires. Met aussi à jour updated_at.
    Exemple: update_object_fields(5, name="Foo", available=0)
    """
    if not fields:
        return
    allowed = {
        "name", "cost_fabrication", "cost_accessory", "cost_total",
        "available", "sold_price", "sold_date", "comment",
        "parent_type", "parent_id"
    }
    sets = []
    params = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        sets.append(f"{k} = ?")
        params.append(v)
    if not sets:
        return

    # Si coûts reçus sans cost_total, on recalcule
    if ("cost_fabrication" in fields or "cost_accessory" in fields) and "cost_total" not in fields:
        cf = float(fields.get("cost_fabrication", 0.0))
        ca = float(fields.get("cost_accessory", 0.0))
        sets.append("cost_total = ?")
        params.append(cf + ca)

    sets.append("updated_at = datetime('now')")

    params.append(int(object_id))

    conn = _connect()
    cur = conn.cursor()
    cur.execute(f"UPDATE objects SET {', '.join(sets)} WHERE id = ?", params)
    conn.commit()
    conn.close()

def set_sold_info(object_id: int, sold_price: Optional[float],
                  sold_date_iso: Optional[str] = None, comment: Optional[str] = None) -> None:
    if sold_date_iso is None:
        # ISO UTC, secondes, suffixe Z (cohérent avec la plupart des parseurs)
        sold_date_iso = datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
    fields = {
        "sold_price": float(sold_price) if sold_price is not None else None,
        "sold_date": sold_date_iso,
        "available": 0,
    }
    if comment is not None:
        fields["comment"] = comment
    update_object_fields(object_id, **fields)


def set_available(object_id: int, available: bool) -> None:
    update_object_fields(object_id, available=1 if available else 0)

def count_existing_objects(source_type: SourceType, source_id: int) -> int:
    conn = _connect(); cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM objects WHERE parent_type=? AND parent_id=?", (source_type, source_id))
    n = int(cur.fetchone()[0]); conn.close()
    return n

def get_available_units(source_type: SourceType, source_id: int) -> int:
    snap = snapshot_source(source_type, source_id)
    already = count_existing_objects(source_type, source_id)
    return max(0, snap.number_of_items - already)

def create_objects_from_source(source_type: SourceType, source_id: int, qty: int) -> int:
    """
    Crée `qty` lignes dans objects :
      - name identique à la source
      - thumbnail identique à la source
      - tags copiés de la source (print: tags du print ; group: union des tags des prints du groupe)
      - cost_fabrication = full_cost unitaire de la source
      - cost_accessory = 0 à la création
      - cost_total = fabrication + accessoires
    Respecte la dispo : max = number_of_items - déjà créés
    """
    if qty <= 0:
        return 0

    snap = snapshot_source(source_type, source_id)
    available = get_available_units(source_type, source_id)
    n = min(qty, available)
    if n <= 0:
        return 0

    conn = _connect(); cur = conn.cursor()
    try:
        conn.execute("BEGIN")

        # On recalcule `already` dans la même transaction pour éviter les courses
        cur.execute("SELECT COUNT(*) FROM objects WHERE parent_type=? AND parent_id=?", (source_type, source_id))
        already = int(cur.fetchone()[0])
        remaining = max(0, snap.number_of_items - already)
        n = min(n, remaining)
        if n <= 0:
            conn.rollback(); return 0

        new_ids: list[int] = []
        for _ in range(n):
            cost_fab = float(snap.full_cost_unit or 0.0)
            cost_acc = 0.0
            cost_total = cost_fab + cost_acc
            cur.execute("""
                INSERT INTO objects (
                    parent_type, parent_id, name, thumbnail,
                    cost_fabrication, cost_accessory, cost_total,
                    available, sold_price, sold_date, comment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, NULL, NULL, NULL)
            """, (source_type, source_id, snap.name, snap.thumbnail,
                  cost_fab, cost_acc, cost_total))
            new_ids.append(int(cur.lastrowid))

        # Copier les tags
        if snap.tags:
            cur.executemany(
                "INSERT OR IGNORE INTO tag_objects (object_id, tag) VALUES (?, ?)",
                [(oid, t) for oid in new_ids for t in snap.tags]
            )

        conn.commit()
        return len(new_ids)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tags (strictement analogues à print_tags)
# ---------------------------------------------------------------------------

def get_object_tags(object_id: int) -> List[str]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT tag FROM tag_objects WHERE object_id = ? ORDER BY LOWER(tag)",
        (int(object_id),),
    )
    tags = [r[0] for r in cur.fetchall()]
    conn.close()
    return tags


def add_object_tag(object_id: int, tag: str) -> None:
    tag = (tag or "").strip()
    if not tag:
        return
    conn = _connect()
    cur = conn.cursor()
    # Idempotent: on ignore si le couple existe déjà
    cur.execute(
        """
        INSERT OR IGNORE INTO tag_objects (object_id, tag)
        VALUES (?, ?)
        """,
        (int(object_id), tag),
    )
    conn.commit()
    conn.close()


def remove_object_tag(object_id: int, tag: str) -> None:
    tag = (tag or "").strip()
    if not tag:
        return
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM tag_objects WHERE object_id = ? AND tag = ?",
        (int(object_id), tag),
    )
    conn.commit()
    conn.close()


def get_tags_for_objects(object_ids: List[int]) -> Dict[int, List[str]]:
    """
    Renvoie un mapping {object_id: [tags]} pour un ensemble d'IDs.
    Utile pour éviter N requêtes côté template.
    """
    if not object_ids:
        return {}
    conn = _connect()
    cur = conn.cursor()
    marks = ",".join("?" for _ in object_ids)
    cur.execute(
        f"SELECT object_id, tag FROM tag_objects WHERE object_id IN ({marks})",
        tuple(int(x) for x in object_ids),
    )
    res: Dict[int, List[str]] = {}
    for oid, tag in cur.fetchall():
        res.setdefault(int(oid), []).append(tag)
    conn.close()
    # tri case-insensitive
    for k in list(res.keys()):
        res[k] = sorted(res[k], key=lambda s: s.lower())
    return res


def list_objects(filters: dict, page: int, per_page: int = 30) -> Tuple[List[Dict], int]:
    """
    Retourne (rows, total_pages) avec filtres simples:
      - search: sous-chaîne dans name (case-insensitive)
      - sold_filter: 'yes' / 'no' / '' (tous)
      - source_type: 'print' / 'group' / '' (tous)
      - available: 'yes' / 'no' / '' (tous)
    """
    clauses = []
    params: list = []

    s = (filters.get("search") or "").strip()
    if s:
        clauses.append("LOWER(name) LIKE ?")
        params.append(f"%{s.lower()}%")

    sf = (filters.get("sold_filter") or "").strip()
    if sf == "yes":
        clauses.append("sold_price IS NOT NULL")
    elif sf == "no":
        clauses.append("sold_price IS NULL")

    st = (filters.get("source_type") or "").strip()
    if st in ("print", "group"):
        clauses.append("parent_type = ?")
        params.append(st)

    av = (filters.get("available") or "").strip()
    if av == "yes":
        clauses.append("available = 1")
    elif av == "no":
        clauses.append("available = 0")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    order_by = "ORDER BY created_at DESC, id DESC"

    conn = _connect()
    cur = conn.cursor()

    # total
    cur.execute(f"SELECT COUNT(*) FROM objects {where}", tuple(params))
    total = int(cur.fetchone()[0]) if cur.fetchone is not None else 0
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    offset = (page - 1) * per_page

    # page
    cur.execute(
        f"""
        SELECT id, name, parent_type, parent_id, thumbnail,
               cost_fabrication, cost_accessory, cost_total,
               available, sold_price, sold_date, comment,
               created_at, updated_at
        FROM objects
        {where}
        {order_by}
        LIMIT ? OFFSET ?
        """,
        tuple(params + [per_page, offset])
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows, total_pages

def rename_object(object_id: int, new_name: str) -> None:
    update_object_fields(int(object_id), name=(new_name or "").strip())

def delete_object(object_id: int) -> None:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM objects WHERE id = ?", (int(object_id),))
    conn.commit()
    conn.close()

ensure_schema()