import sqlite3
from typing import Optional, Dict, Any, List, Tuple, NamedTuple, Literal, TypedDict
from datetime import datetime, timezone
from collections.abc import Iterable

# On réutilise la config DB telle qu'elle existe déjà dans le projet
from print_history import db_config

def _normalize_sale_filter(filters: dict) -> str:
    """
    Harmonise la valeur du filtre 'Vente' en une des valeurs:
      'vendus' | 'dispo' | 'offert' | 'perso' | ''.
    Rétro-compatibilité:
      - sold_filter=yes  -> 'vendus'
      - available=yes    -> 'dispo'
      - sold_filter=no   -> '' (pas de contrainte)
      - available=no     -> '' (pas de contrainte)
    """
    val = (filters.get("sale_filter") or "").strip().lower()

    if not val:
        sf = (filters.get("sold_filter") or "").strip().lower()
        av = (filters.get("available") or "").strip().lower()
        if sf == "yes": val = "vendus"
        elif av == "yes": val = "dispo"
        elif sf == "no" or av == "no": val = ""
        else: val = ""

    # alias tolérés
    if val in ("sold", "payes", "payés"):
        val = "vendus"
    if val in ("available", "disponible", "disponibles"):
        val = "dispo"
    if val in ("gifted", "don", "dons"):
        val = "offert"
    if val in ("perso", "personnel", "personnels", "personal"):
        val = "perso"

    return val


def _build_objects_where(filters: dict):
    """
    Construit (where_sql, params) à partir des filtres: search, source_type, sale_filter.
    """
    clauses = []
    params = []

    s = (filters.get("search") or "").strip()
    if s:
        clauses.append("LOWER(name) LIKE ?")
        params.append(f"%{s.lower()}%")

    st = (filters.get("source_type") or "").strip()
    if st in ("print", "group"):
        clauses.append("parent_type = ?")
        params.append(st)

    sale = _normalize_sale_filter(filters)
    if sale == "vendus":
        clauses.append("sold_price > 0")
    elif sale == "dispo":
        clauses.append("available = 1")
    elif sale == "offert":
        clauses.append("(sold_price = 0 AND personal = 0)")
    elif sale == "perso":
        clauses.append("(sold_price = 0 AND personal = 1)")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, tuple(params)


def _extend_where(base_where: str, extra: str) -> str:
    return f"{base_where} AND {extra}" if base_where else f"WHERE {extra}"

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
            translated_name  TEXT    NOT NULL,
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
    # Table des groupes d'objets
    cur.execute("""
        CREATE TABLE IF NOT EXISTS object_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    cur.execute("PRAGMA table_info(objects)")
    cols = {r[1] for r in cur.fetchall()}
    if "thumbnail" not in cols:
        cur.execute("ALTER TABLE objects ADD COLUMN thumbnail TEXT")
    if "translated_name" not in cols:
        cur.execute("ALTER TABLE objects ADD COLUMN translated_name TEXT")
    if "personal" not in cols:
        cur.execute("ALTER TABLE objects ADD COLUMN personal INTEGER NOT NULL DEFAULT 0")
    if "object_group_id" not in cols:
        cur.execute("ALTER TABLE objects ADD COLUMN object_group_id INTEGER NULL")
    if "normal_cost_unit" not in cols:
        cur.execute("ALTER TABLE objects ADD COLUMN normal_cost_unit REAL")
    cur.execute("""
    UPDATE objects
    SET normal_cost_unit = COALESCE(
            (SELECT CASE
                    WHEN p.full_normal_cost_by_item IS NOT NULL
                            THEN p.full_normal_cost_by_item
                    WHEN p.full_normal_cost IS NOT NULL
                            AND COALESCE(p.number_of_items,1) > 0
                            THEN p.full_normal_cost / COALESCE(p.number_of_items,1)
                    ELSE NULL
                    END
                FROM prints p
            WHERE p.id = objects.parent_id),
            normal_cost_unit
        ),
        updated_at = datetime('now')
    WHERE parent_type = 'print'
    AND (normal_cost_unit IS NULL OR normal_cost_unit = 0);
    """)
    
    # --- Backfill des enregistrements existants (group) ---
    cur.execute("""
    UPDATE objects
    SET normal_cost_unit = COALESCE(
            (SELECT CASE
                    WHEN g.full_normal_cost_by_item IS NOT NULL
                            THEN g.full_normal_cost_by_item
                    WHEN g.full_normal_cost IS NOT NULL
                            AND COALESCE(g.number_of_items,1) > 0
                            THEN g.full_normal_cost / COALESCE(g.number_of_items,1)
                    ELSE NULL
                    END
                FROM print_groups g
            WHERE g.id = objects.parent_id),
            normal_cost_unit
        ),
        updated_at = datetime('now')
    WHERE parent_type = 'group'
    AND (normal_cost_unit IS NULL OR normal_cost_unit = 0);
    """)
    if "margin" not in cols:
        cur.execute("ALTER TABLE objects ADD COLUMN margin REAL")
        # backfill initial : calcule la marge existante si sold_price défini
        cur.execute("""
            UPDATE objects
            SET margin = CASE
                WHEN sold_price IS NOT NULL THEN
                    sold_price - COALESCE(cost_total, COALESCE(cost_accessory,0) + COALESCE(cost_fabrication,0))
                ELSE NULL
            END
        """)
    cur.execute("DROP TRIGGER IF EXISTS trg_sync_obj_cost_after_print_update")
    cur.execute("DROP TRIGGER IF EXISTS trg_sync_obj_cost_after_group_update")
    cur.execute("DROP TRIGGER IF EXISTS trg_obj_sync_after_print_cost")
    cur.execute("DROP TRIGGER IF EXISTS trg_obj_sync_after_group_cost")
    cur.execute("DROP TRIGGER IF EXISTS trg_obj_recompute_after_components")
    cur.execute("DROP TRIGGER IF EXISTS trg_obj_recompute_after_sold_price")
    cur.execute("DROP TRIGGER IF EXISTS trg_obj_sync_norm_after_print_update")
    cur.execute("DROP TRIGGER IF EXISTS trg_obj_sync_norm_after_group_update")
    cur.execute("""
        SELECT name FROM sqlite_master
         WHERE type='trigger' AND name IN (
            'trg_objects_group_cleanup_after_update',
            'trg_objects_group_cleanup_after_delete'
         )
    """)
    existing_triggers = {r[0] for r in cur.fetchall()}

    # 1) Quand on change le groupement d'un objet (UPDATE object_group_id),
    #    si l'ANCIEN groupe devient vide, on le supprime.
    if 'trg_objects_group_cleanup_after_update' not in existing_triggers:
        cur.execute("""
            CREATE TRIGGER trg_objects_group_cleanup_after_update
            AFTER UPDATE OF object_group_id ON objects
            WHEN OLD.object_group_id IS NOT NULL
            BEGIN
                DELETE FROM object_groups
                 WHERE id = OLD.object_group_id
                   AND NOT EXISTS (
                        SELECT 1 FROM objects
                         WHERE object_group_id = OLD.object_group_id
                   );
            END;
        """)

    # 2) Quand on supprime un objet (DELETE),
    #    si son groupe devient vide, on le supprime.
    if 'trg_objects_group_cleanup_after_delete' not in existing_triggers:
        cur.execute("""
            CREATE TRIGGER trg_objects_group_cleanup_after_delete
            AFTER DELETE ON objects
            WHEN OLD.object_group_id IS NOT NULL
            BEGIN
                DELETE FROM object_groups
                 WHERE id = OLD.object_group_id
                   AND NOT EXISTS (
                        SELECT 1 FROM objects
                         WHERE object_group_id = OLD.object_group_id
                   );
            END;
    """)

    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS trg_obj_sync_norm_after_print_update
    AFTER UPDATE OF full_normal_cost_by_item, full_normal_cost, number_of_items ON prints
    BEGIN
        UPDATE objects
        SET normal_cost_unit = COALESCE(
                    CASE
                        WHEN NEW.full_normal_cost_by_item IS NOT NULL THEN NEW.full_normal_cost_by_item
                        WHEN NEW.full_normal_cost IS NOT NULL AND COALESCE(NEW.number_of_items,1) > 0
                            THEN NEW.full_normal_cost / COALESCE(NEW.number_of_items,1)
                        ELSE normal_cost_unit
                    END, normal_cost_unit),
            updated_at = datetime('now')
        WHERE parent_type = 'print' AND parent_id = NEW.id;
    END;
    """)
    
    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS trg_obj_sync_norm_after_group_update
    AFTER UPDATE OF full_normal_cost_by_item, full_normal_cost, number_of_items ON print_groups
    BEGIN
        UPDATE objects
        SET normal_cost_unit = COALESCE(
                    CASE
                        WHEN NEW.full_normal_cost_by_item IS NOT NULL THEN NEW.full_normal_cost_by_item
                        WHEN NEW.full_normal_cost IS NOT NULL AND COALESCE(NEW.number_of_items,1) > 0
                            THEN NEW.full_normal_cost / COALESCE(NEW.number_of_items,1)
                        ELSE normal_cost_unit
                    END, normal_cost_unit),
            updated_at = datetime('now')
        WHERE parent_type = 'group' AND parent_id = NEW.id;
    END;
    """)

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
     # --- helper SQL commun : coût unitaire de la source -------------------
    # unit = NEW.full_cost_by_item
    #      ou (NEW.full_cost si number_of_items=1)
    #      sinon NULL (on ne change pas le coût côté objets)
    unit_sql = """
        COALESCE(
            NEW.full_cost_by_item,
            CASE WHEN COALESCE(NEW.number_of_items,1)=1 THEN COALESCE(NEW.full_cost,0) END
        )
    """

    # --- PRINTS : recalc des objets liés quand coûts changent -------------
    cur.execute(f"""
        CREATE TRIGGER IF NOT EXISTS trg_obj_sync_after_print_cost
        AFTER UPDATE OF full_cost_by_item, full_cost, number_of_items ON prints
        WHEN
              (NEW.full_cost_by_item IS NOT OLD.full_cost_by_item)
           OR (COALESCE(NEW.number_of_items,1)=1 AND NEW.full_cost IS NOT OLD.full_cost)
           OR (COALESCE(NEW.number_of_items,1) != COALESCE(OLD.number_of_items,1))
        BEGIN
            UPDATE objects
            SET
                cost_fabrication = COALESCE({unit_sql}, cost_fabrication),
                cost_total       = COALESCE(cost_accessory,0) + COALESCE({unit_sql}, cost_fabrication),
                margin           = CASE
                                      WHEN sold_price IS NOT NULL THEN
                                          sold_price - (COALESCE(cost_accessory,0) + COALESCE({unit_sql}, cost_fabrication))
                                      ELSE NULL
                                   END,
                updated_at       = datetime('now')
            WHERE parent_type='print' AND parent_id=NEW.id;
        END;
    """)

    # --- GROUPES : recalc des objets liés quand coûts changent ------------
    cur.execute(f"""
        CREATE TRIGGER IF NOT EXISTS trg_obj_sync_after_group_cost
        AFTER UPDATE OF full_cost_by_item, full_cost, number_of_items ON print_groups
        WHEN
              (NEW.full_cost_by_item IS NOT OLD.full_cost_by_item)
           OR (COALESCE(NEW.number_of_items,1)=1 AND NEW.full_cost IS NOT OLD.full_cost)
           OR (COALESCE(NEW.number_of_items,1) != COALESCE(OLD.number_of_items,1))
        BEGIN
            UPDATE objects
            SET
                cost_fabrication = COALESCE({unit_sql}, cost_fabrication),
                cost_total       = COALESCE(cost_accessory,0) + COALESCE({unit_sql}, cost_fabrication),
                margin           = CASE
                                      WHEN sold_price IS NOT NULL THEN
                                          sold_price - (COALESCE(cost_accessory,0) + COALESCE({unit_sql}, cost_fabrication))
                                      ELSE NULL
                                   END,
                updated_at       = datetime('now')
            WHERE parent_type='group' AND parent_id=NEW.id;
        END;
    """)

    # --- OBJECTS : garder cost_total & margin alignés quand on modifie les composants
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_obj_recompute_after_components
        AFTER UPDATE OF cost_accessory, cost_fabrication ON objects
        BEGIN
            UPDATE objects
            SET
                cost_total = COALESCE(NEW.cost_accessory,0) + COALESCE(NEW.cost_fabrication,0),
                margin     = CASE
                                WHEN NEW.sold_price IS NOT NULL
                                  THEN NEW.sold_price - (COALESCE(NEW.cost_accessory,0) + COALESCE(NEW.cost_fabrication,0))
                                ELSE NULL
                             END,
                updated_at = datetime('now')
            WHERE id = NEW.id;
        END;
    """)

    # --- OBJECTS : recalculer margin quand sold_price change --------------
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_obj_recompute_after_sold_price
        AFTER UPDATE OF sold_price ON objects
        BEGIN
            UPDATE objects
            SET
                margin     = CASE
                                WHEN NEW.sold_price IS NOT NULL
                                  THEN NEW.sold_price - COALESCE(NEW.cost_total, COALESCE(NEW.cost_accessory,0)+COALESCE(NEW.cost_fabrication,0))
                                ELSE NULL
                             END,
                updated_at = datetime('now')
            WHERE id = NEW.id;
        END;
    """)
    ensure_accessories_schema(cur)
    conn.commit()
    conn.close()

def ensure_accessories_schema(cur: sqlite3.Cursor) -> None:
    """
    Crée/upgrade les tables 'accessories' et 'object_accessories' + index + triggers.
    Appelée depuis ensure_schema().
    """
    # Table des accessoires (stock global)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS accessories (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL UNIQUE,
            quantity    INTEGER NOT NULL DEFAULT 0,
            unit_price  REAL    NOT NULL DEFAULT 0,  -- prix unitaire actuel (moyenne pondérée)
            image_path  TEXT,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_accessories_name ON accessories(name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_accessories_qty  ON accessories(quantity)")

    # Liaisons Objet <-> Accessoire (prix unitaire figé au moment du lien)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS object_accessories (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            object_id            INTEGER NOT NULL REFERENCES objects(id)    ON DELETE CASCADE,
            accessory_id         INTEGER NOT NULL REFERENCES accessories(id) ON DELETE CASCADE,
            quantity             INTEGER NOT NULL CHECK (quantity > 0),
            unit_price_at_link   REAL    NOT NULL,  -- snapshot : ne change plus après l’affectation
            created_at           TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(object_id, accessory_id)        -- 1 ligne par (objet, accessoire), on cumule dans quantity
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_objacc_object ON object_accessories(object_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_objacc_acc    ON object_accessories(accessory_id)")

    # --- TRIGGERS ---

    # Après INSERT : décrémente le stock accessoire + recalcule cost_accessory de l'objet
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_objacc_after_insert
        AFTER INSERT ON object_accessories
        BEGIN
            UPDATE accessories
               SET quantity = quantity - NEW.quantity,
                   updated_at = datetime('now')
             WHERE id = NEW.accessory_id;

            UPDATE objects
               SET cost_accessory = COALESCE((
                        SELECT SUM(quantity * unit_price_at_link)
                          FROM object_accessories
                         WHERE object_id = NEW.object_id
                    ),0),
                   updated_at = datetime('now')
             WHERE id = NEW.object_id;
        END;
    """)

    # Après UPDATE de quantity : ajuste le stock par delta + recalcule cost_accessory
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_objacc_after_update
        AFTER UPDATE OF quantity ON object_accessories
        BEGIN
            UPDATE accessories
               SET quantity = quantity - (NEW.quantity - OLD.quantity),
                   updated_at = datetime('now')
             WHERE id = NEW.accessory_id;

            UPDATE objects
               SET cost_accessory = COALESCE((
                        SELECT SUM(quantity * unit_price_at_link)
                          FROM object_accessories
                         WHERE object_id = NEW.object_id
                    ),0),
                   updated_at = datetime('now')
             WHERE id = NEW.object_id;
        END;
    """)

    # Après DELETE : recrédite le stock + recalcule cost_accessory
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_objacc_after_delete
        AFTER DELETE ON object_accessories
        BEGIN
            UPDATE accessories
               SET quantity = quantity + OLD.quantity,
                   updated_at = datetime('now')
             WHERE id = OLD.accessory_id;

            UPDATE objects
               SET cost_accessory = COALESCE((
                        SELECT SUM(quantity * unit_price_at_link)
                          FROM object_accessories
                         WHERE object_id = OLD.object_id
                    ),0),
                   updated_at = datetime('now')
             WHERE id = OLD.object_id;
        END;
    """)

    # 📌 NB : ton trigger existant 'trg_obj_recompute_after_components' sur objects
    # recalculera cost_total/margin quand cost_accessory bouge — pas besoin d’en rajouter.


# =========================
#  ACCESSOIRES - DAL (API Python)
# =========================

def list_accessories() -> list[dict]:
    conn = _connect(); cur = conn.cursor()
    cur.execute("""
        SELECT id, name, quantity, unit_price, image_path, created_at, updated_at
          FROM accessories
         ORDER BY name COLLATE NOCASE
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def set_accessory_image_path(acc_id: int, image_path: str | None) -> None:
    """
    Met à jour le chemin du visuel d'un accessoire.
    - image_path: chemin relatif (ex: 'uploads/accessories/acc_12_xxx.png') ou None
    Lève ValueError si l'accessoire n'existe pas.
    """
    conn = _connect(); cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE accessories SET image_path=?, updated_at=datetime('now') WHERE id=?",
            (image_path, acc_id)
        )
        if cur.rowcount == 0:
            raise ValueError("Accessoire introuvable")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def get_accessory(acc_id: int) -> dict | None:
    conn = _connect(); cur = conn.cursor()
    cur.execute("""
        SELECT id, name, quantity, unit_price, image_path, created_at, updated_at
          FROM accessories
         WHERE id = ?
    """, (acc_id,))
    r = cur.fetchone()
    conn.close()
    return dict(r) if r else None

def create_accessory(name: str, qty: int, total_price: float, image_path: str | None = None) -> int:
    """
    Crée un accessoire avec stock initial.
    - unit_price = total_price / qty (0 si qty=0)
    """
    if not name or qty < 0 or total_price < 0:
        raise ValueError("Paramètres invalides pour create_accessory")
    unit_price = (float(total_price) / qty) if qty > 0 else 0.0

    conn = _connect(); cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO accessories(name, quantity, unit_price, image_path)
                 VALUES (?, ?, ?, ?)
        """, (name.strip(), int(qty), float(unit_price), image_path))
        acc_id = cur.lastrowid
        conn.commit()
        return int(acc_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def add_accessory_stock(acc_id: int, add_qty: int, add_total_price: float) -> None:
    """
    Ajoute du stock à un accessoire existant en recalculant le prix unitaire
    par moyenne pondérée :
      new_PU = (old_qty*old_PU + add_total_price) / (old_qty + add_qty)
    """
    if add_qty <= 0 or add_total_price < 0:
        raise ValueError("add_qty doit être > 0 et add_total_price >= 0")

    conn = _connect(); cur = conn.cursor()
    try:
        cur.execute("SELECT quantity, unit_price FROM accessories WHERE id=?", (acc_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError("Accessoire introuvable")

        old_qty = int(row["quantity"] or 0)
        old_pu  = float(row["unit_price"] or 0.0)
        new_qty = old_qty + int(add_qty)
        new_pu  = (old_qty * old_pu + float(add_total_price)) / new_qty if new_qty > 0 else 0.0

        cur.execute("""
            UPDATE accessories
               SET quantity = ?, unit_price = ?, updated_at = datetime('now')
             WHERE id = ?
        """, (new_qty, new_pu, acc_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def remove_accessory_stock(acc_id: int, remove_qty: int) -> None:
    """
    Retire du stock à un accessoire sans changer le prix unitaire.
    - remove_qty doit être > 0 et <= quantité actuelle.
    """
    if remove_qty <= 0:
        raise ValueError("remove_qty doit être > 0")

    conn = _connect(); cur = conn.cursor()
    try:
        cur.execute("SELECT quantity FROM accessories WHERE id=?", (acc_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError("Accessoire introuvable")

        current = int(row["quantity"] or 0)
        if remove_qty > current:
            raise ValueError("Quantité à retirer supérieure au stock disponible")

        new_qty = current - remove_qty
        cur.execute("""
            UPDATE accessories
               SET quantity=?, updated_at=datetime('now')
             WHERE id=?
        """, (new_qty, acc_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def delete_accessory(acc_id: int) -> None:
    """
    Supprime l'accessoire. Les liaisons object_accessories sont supprimées
    grâce au ON DELETE CASCADE.
    """
    conn = _connect(); cur = conn.cursor()
    try:
        cur.execute("DELETE FROM accessories WHERE id=?", (acc_id,))
        if cur.rowcount == 0:
            raise ValueError("Accessoire introuvable")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def link_accessory_to_object(object_id: int, accessory_id: int, qty: int) -> int:
    """
    Lie un accessoire à un objet (et décrémente le stock).
    - Snapshot du prix unitaire à l’instant T (unit_price_at_link)
    - Si (object_id, accessory_id) existe : on incrémente quantity (UPSERT)
    Retourne l'id de la ligne de liaison.
    """
    if qty <= 0:
        raise ValueError("La quantité doit être > 0")

    conn = _connect(); cur = conn.cursor()
    try:
        # Vérification du stock dispo et récupération du PU courant
        cur.execute("SELECT quantity, unit_price FROM accessories WHERE id=?", (accessory_id,))
        acc = cur.fetchone()
        if not acc:
            raise ValueError("Accessoire introuvable")
        if int(acc["quantity"]) < int(qty):
            raise ValueError("Stock insuffisant")

        unit_price = float(acc["unit_price"] or 0.0)

        # UPSERT grâce à UNIQUE(object_id, accessory_id)
        cur.execute("""
            INSERT INTO object_accessories(object_id, accessory_id, quantity, unit_price_at_link)
                 VALUES (?, ?, ?, ?)
            ON CONFLICT(object_id, accessory_id) DO UPDATE SET
                 quantity = quantity + excluded.quantity
        """, (int(object_id), int(accessory_id), int(qty), unit_price))

        # Récup id de la ligne
        cur.execute("""
            SELECT id FROM object_accessories
             WHERE object_id=? AND accessory_id=?
        """, (object_id, accessory_id))
        row = cur.fetchone()
        conn.commit()
        return int(row["id"])
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def unlink_accessory_from_object(object_id: int, accessory_id: int, qty: int | None = None) -> None:
    """
    Supprime tout ou partie du lien d'un accessoire avec un objet, en rendant le stock.
    - qty=None  => supprime entièrement la ligne
    - qty>0     => décrémente la quantité ; si elle atteint 0, supprime la ligne
    Les triggers se chargent de réajuster le stock accessoire et le coût objet.
    """
    conn = _connect(); cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, quantity FROM object_accessories
             WHERE object_id=? AND accessory_id=?
        """, (object_id, accessory_id))
        row = cur.fetchone()
        if not row:
            return  # rien à faire

        if qty is None or int(qty) >= int(row["quantity"]):
            cur.execute("DELETE FROM object_accessories WHERE id=?", (row["id"],))
        else:
            new_q = int(row["quantity"]) - int(qty)
            cur.execute("UPDATE object_accessories SET quantity=? WHERE id=?", (new_q, row["id"]))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def list_object_accessories(object_id: int) -> list[dict]:
    """
    Renvoie la liste des accessoires liés à un objet (id accessoire, nom, image, qté, PU snapshot, coût total).
    """
    conn = _connect(); cur = conn.cursor()
    cur.execute("""
        SELECT oa.id,
               oa.accessory_id,
               a.name AS accessory_name,
               a.image_path AS image_path,
               oa.quantity,
               oa.unit_price_at_link,
               (oa.quantity * oa.unit_price_at_link) AS total_cost
          FROM object_accessories oa
          JOIN accessories a ON a.id = oa.accessory_id
         WHERE oa.object_id = ?
         ORDER BY a.name COLLATE NOCASE
    """, (object_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

# ---------------------------------------------------------------------------
# Helpers : lecture / écriture simples
# ---------------------------------------------------------------------------

SourceType = Literal["print", "group"]

class SourceSnapshot(NamedTuple):
    name: str
    translated_name: str
    number_of_items: int
    full_cost_unit: float
    normal_cost_unit: float 
    thumbnail: str | None
    tags: list[str]
    created_at: str | None  # 👈 date d’origine (print_date ou dernier print du groupe)

def _compute_full_unit(units: int, full_cost_by_item, full_cost) -> float:
    try:
        if full_cost_by_item is not None:
            return float(full_cost_by_item)
        if full_cost is not None and units:
            return float(full_cost) / max(1, int(units))
    except Exception:
        pass
    return 0.0

def _compute_normal_unit(units: int, normal_by_item, normal_total) -> float:
    try:
        if normal_by_item is not None:
            return float(normal_by_item)
        if normal_total is not None and units:
            return float(normal_total) / max(1, int(units))
    except Exception:
        pass
    return 0.0

def _snapshot_from_print(print_id: int) -> SourceSnapshot:
    conn = _connect(); cur = conn.cursor()
    cur.execute("""
        SELECT
            id,
            COALESCE(file_name, 'Print #' || id) AS name,
            COALESCE(number_of_items, 1)        AS number_of_items,
            translated_name,
            full_cost_by_item,
            full_cost,
            full_normal_cost_by_item,
            full_normal_cost,
            image_file,
            print_date
        FROM prints
        WHERE id = ?
    """, (print_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return SourceSnapshot(f"Print #{print_id}", f"Print #{print_id}", 1, 0.0, 0.0, None, [], None)

    units = int(row["number_of_items"] or 1)
    full_unit = _compute_full_unit(units, row["full_cost_by_item"], row["full_cost"])
    normal_unit = _compute_normal_unit(units, row["full_normal_cost_by_item"], row["full_normal_cost"])
    thumb = f"/static/prints/{row['image_file']}" if row["image_file"] else None
    created_at = row["print_date"]  # on prend la date du print telle quelle

    # Tags du print
    cur.execute("SELECT tag FROM print_tags WHERE print_id = ? ORDER BY LOWER(tag)", (print_id,))
    tags = [r[0] for r in cur.fetchall()]
    conn.close()

    return SourceSnapshot(row["name"],row["translated_name"], max(1, units), full_unit, normal_unit, thumb, tags, created_at)

def _snapshot_from_group(group_id: int) -> SourceSnapshot:
    conn = _connect(); cur = conn.cursor()
    cur.execute("""
        SELECT
            id,
            COALESCE(name, 'Groupe ' || id) AS name,
            COALESCE(number_of_items, 1)    AS number_of_items,
            full_cost_by_item,
            full_cost,
            full_normal_cost_by_item,
            full_normal_cost,
            primary_print_id
        FROM print_groups
        WHERE id = ?
    """, (group_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return SourceSnapshot(f"Groupe #{group_id}", f"Groupe #{group_id}", 1, 0.0, 0.0, None, [], None)

    units = int(row["number_of_items"] or 1)
    full_unit = _compute_full_unit(units, row["full_cost_by_item"], row["full_cost"])
    normal_unit = _compute_normal_unit(units, row["full_normal_cost_by_item"], row["full_normal_cost"])

    # Thumbnail : priorité au print de référence
    thumb = None
    if row["primary_print_id"]:
        cur.execute("SELECT image_file FROM prints WHERE id = ?", (row["primary_print_id"],))
        p = cur.fetchone()
        if p and p["image_file"]:
            thumb = f"/static/prints/{p['image_file']}"

    # Fallback thumbnail + created_at : dernier print du groupe par date
    cur.execute("""
        SELECT image_file, print_date
        FROM prints
        WHERE group_id = ?
        ORDER BY datetime(print_date) DESC, id DESC
        LIMIT 1
    """, (group_id,))
    lastp = cur.fetchone()
    created_at = None
    if lastp:
        created_at = lastp["print_date"]
        if not thumb and lastp["image_file"]:
            thumb = f"/static/prints/{lastp['image_file']}"

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
    return SourceSnapshot(row["name"],row["name"], max(1, units), full_unit,normal_unit, thumb, tags, created_at)

def snapshot_source(source_type: SourceType, source_id: int) -> SourceSnapshot:
    if source_type == "print":
        return _snapshot_from_print(source_id)
    elif source_type == "group":
        return _snapshot_from_group(source_id)
    else:
        return SourceSnapshot(f"{source_type} #{source_id}", 1, 0.0, None, [], None)
    
def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(row) if row else {}


def create_object(
    parent_type: str,
    parent_id: int,
    name: str,
    translated_name: str,
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
            parent_type, parent_id, name,translated_name
            cost_fabrication, cost_accessory, cost_total,
            available, sold_price, sold_date, comment
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            parent_type, int(parent_id), name.strip(),translated_name.strip(),
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
    Crée `qty` objets à partir d'une source (print|group), en respectant la dispo :
      dispo = number_of_items(source) - objets déjà créés pour (type,id)
    Chaque objet créé reçoit :
      - name           = nom de la source
      - thumbnail      = vignette de la source (print.image_file ; pour group: primary print
                         sinon dernier print par date)
      - cost_fabrication = full_cost unitaire de la source
      - cost_accessory   = 0 à la création
      - cost_total       = fabrication + accessoires
      - created_at / updated_at = date d’origine de la source (print_date pour print ;
                         pour group: date du dernier print du groupe). Si inconnue, on laisse
                         les DEFAULT SQL (datetime('now')).
      - tags            = copie des tags de la source (print_tags ; pour group: union des tags
                         des prints du groupe)
    Retourne le nombre d’objets effectivement créés (0 si rien à créer).
    """
    if source_type not in ("print", "group"):
        return 0
    qty = int(qty or 0)
    if qty <= 0:
        return 0

    snap = snapshot_source(source_type, source_id)

    conn = _connect()
    cur = conn.cursor()
    try:
        conn.execute("BEGIN")

        # Recompte dans la transaction pour éviter les conditions de course
        cur.execute(
            "SELECT COUNT(*) FROM objects WHERE parent_type=? AND parent_id=?",
            (source_type, source_id)
        )
        already = int(cur.fetchone()[0])
        remaining = max(0, int(snap.number_of_items) - already)
        n = min(qty, remaining)
        if n <= 0:
            conn.rollback()
            return 0

        # Valeurs unitaires constantes pour cette salve
        cost_fab = float(snap.full_cost_unit or 0.0)
        cost_acc = 0.0
        cost_total = cost_fab + cost_acc
        normal_cost_unit = float(snap.normal_cost_unit or 0.0)
        created_iso = snap.created_at  # peut être None
        new_ids: list[int] = []

        if created_iso is None:
            # On laisse SQLite remplir created_at/updated_at via les DEFAULT
            for _ in range(n):
                cur.execute(
                    """
                    INSERT INTO objects (
                        parent_type, parent_id, name,translated_name, thumbnail,
                        cost_fabrication, cost_accessory, cost_total,normal_cost_unit, 
                        available, sold_price, sold_date, comment
                    )
                    VALUES (?, ?, ?, ?,?, ?, ?, ?,?, 1, NULL, NULL, NULL)
                    """,
                    (source_type, source_id, snap.name,snap.translated_name, snap.thumbnail,
                     cost_fab, cost_acc, cost_total,normal_cost_unit)
                )
                new_ids.append(int(cur.lastrowid))
        else:
            # On force created_at / updated_at = date d’origine
            for _ in range(n):
                cur.execute(
                    """
                    INSERT INTO objects (
                        parent_type, parent_id, name,translated_name, thumbnail,
                        cost_fabrication, cost_accessory, cost_total,normal_cost_unit,
                        available, sold_price, sold_date, comment,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?,?, ?, ?,?, 1, NULL, NULL, NULL, ?, ?)
                    """,
                    (source_type, source_id, snap.name,snap.translated_name, snap.thumbnail,
                     cost_fab, cost_acc, cost_total,normal_cost_unit,
                     created_iso, created_iso)
                )
                new_ids.append(int(cur.lastrowid))

        # Copie des tags
        if snap.tags and new_ids:
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

def get_object_counts_by_parent(parent_type: str, parent_ids: Iterable[int]) -> dict[int, int]:
    ids = [int(x) for x in set(parent_ids) if x]
    if not ids:
        return {}
    conn = _connect(); cur = conn.cursor()
    marks = ",".join("?" for _ in ids)
    cur.execute(
        f"""SELECT parent_id, COUNT(*) AS n
            FROM objects
            WHERE parent_type = ? AND parent_id IN ({marks})
            GROUP BY parent_id""",
        tuple([parent_type] + ids)
    )
    out: Dict[int, int] = {int(r[0]): int(r[1]) for r in cur.fetchall()}
    conn.close()
    return out


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

def list_objects(filters: dict, page: int, per_page: int = 30):
    """
    Retourne une page d'objets filtrés avec total_pages.
    """
    where, params = _build_objects_where(filters)

    conn = _connect()
    cur = conn.cursor()

    # Compter le total
    cur.execute(f"SELECT COUNT(*) FROM objects {where}", params)
    total_count = cur.fetchone()[0]
    total_pages = max(1, (total_count + per_page - 1) // per_page)

    # Récupérer les lignes (triées par date desc si dispo)
    offset = (page - 1) * per_page
    cur.execute(f"""
        SELECT *
        FROM objects
        {where}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    """, params + (per_page, offset))
    rows = cur.fetchall()

    conn.close()
    return rows, total_pages

def list_objects_using_accessory(accessory_id: int) -> list[dict]:
    """
    Renvoie les objets qui utilisent l'accessoire donné, avec qté, thumbnail et référence source.
    """
    conn = _connect(); cur = conn.cursor()
    cur.execute("""
        SELECT
            o.id               AS object_id,
            o.name             AS object_name,
            o.parent_type,
            o.parent_id,
            oa.quantity        AS quantity,
            o.thumbnail        AS thumbnail
        FROM object_accessories oa
        JOIN objects o ON o.id = oa.object_id
        WHERE oa.accessory_id = ?
        ORDER BY o.created_at DESC, o.id DESC
    """, (accessory_id,))
    out = [dict(r) for r in cur.fetchall()]
    conn.close()
    return out

def rename_accessory(acc_id: int, new_name: str):
    conn = _connect(); cur = conn.cursor()
    cur.execute("UPDATE accessories SET name=? WHERE id=?", (new_name, acc_id))
    conn.commit(); conn.close()

def rename_object(object_id: int, new_name: str) -> None:
    update_object_fields(int(object_id), name=(new_name or "").strip())

def delete_object(object_id: int) -> None:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM objects WHERE id = ?", (int(object_id),))
    conn.commit()
    conn.close()

def update_object_sale(object_id: int, sold_price: float, sold_date: str, comment: Optional[str], sold_personal: bool = False) -> None:
    """
    Enregistre la vente / don d'un objet.
      - sold_price >= 0, 0 = don ou personnel (selon sold_personal)
      - sold_personal True => objet personnel (uniquement si sold_price == 0)
    Effets :
      - available = 0
      - sold_price, sold_date, personal, comment, updated_at
    """
    conn = _connect(); cur = conn.cursor()

    personal_val = 1 if (sold_price == 0 and sold_personal) else 0

    cur.execute(
        """
        UPDATE objects
           SET sold_price = ?,
               sold_date  = ?,
               personal   = ?,
               comment    = ?,
               available  = 0,
               updated_at = datetime('now')
         WHERE id = ?
        """,
        (sold_price, sold_date, personal_val, comment, object_id),
    )
    if cur.rowcount == 0:
        conn.rollback(); conn.close()
        raise ValueError(f"Objet introuvable (id={object_id})")
    conn.commit(); conn.close()

def clear_object_sale(object_id: int) -> None:
    """
    Annule la vente/don d'un objet :
      - sold_price = NULL
      - sold_date  = NULL
      - available  = 1 (remis en stock)
      - updated_at = now
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE objects
            SET sold_price = NULL,
                sold_date  = NULL,
                personal   = 0,
                available  = 1,
                updated_at = datetime('now')
        WHERE id = ?
        """,
        (object_id,),
    )
    if cur.rowcount == 0:
        conn.rollback()
        conn.close()
        raise ValueError(f"Objet introuvable (id={object_id})")

    conn.commit()
    conn.close()

def update_object_comment(object_id: int, comment: Optional[str]) -> None:
    """
    Met à jour (ou efface) le commentaire d'un objet.
    - comment=None => efface le commentaire
    - met à jour updated_at = now
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE objects
           SET comment    = ?,
               updated_at = datetime('now')
         WHERE id = ?
        """,
        (comment, object_id),
    )
    if cur.rowcount == 0:
        conn.rollback(); conn.close()
        raise ValueError(f"Objet introuvable (id={object_id})")
    conn.commit(); conn.close()

class ObjectsSummary(TypedDict):
    total_objects: int
    sold_count: int
    available_count: int
    gifted_count: int      # sold_price = 0 AND personal = 0
    personal_count: int    # sold_price = 0 AND personal = 1
    sum_sold_price: float
    sum_positive_margin: float

def summarize_objects(filters: dict) -> ObjectsSummary:
    """
    Calcule des agrégats sur la table objects selon les filtres fournis.
    """
    where, params = _build_objects_where(filters)

    conn = _connect()
    cur = conn.cursor()

    # Total
    cur.execute(f"SELECT COUNT(*) FROM objects {where}", params)
    total_objects = int(cur.fetchone()[0])

    # Vendus (sold_price > 0)
    cur.execute(f"SELECT COUNT(*) FROM objects {_extend_where(where, 'sold_price > 0')}", params)
    sold_count = int(cur.fetchone()[0])

    # Disponibles
    cur.execute(f"SELECT COUNT(*) FROM objects {_extend_where(where, 'available = 1')}", params)
    available_count = int(cur.fetchone()[0])

    # Offerts (=0 & personal=0)
    cur.execute(f"SELECT COUNT(*) FROM objects {_extend_where(where, '(sold_price = 0 AND personal = 0)')}", params)
    gifted_count = int(cur.fetchone()[0])

    # Perso (=0 & personal=1)
    cur.execute(f"SELECT COUNT(*) FROM objects {_extend_where(where, '(sold_price = 0 AND personal = 1)')}", params)
    personal_count = int(cur.fetchone()[0])

    # Somme des prix de vente (NULL exclus, inclut 0 et >0)
    cur.execute(f"""
        SELECT COALESCE(SUM(sold_price), 0)
        FROM objects
        {_extend_where(where, 'sold_price IS NOT NULL')}
    """, params)
    sum_sold_price = float(cur.fetchone()[0] or 0.0)

    # Somme des marges positives (ignore dons et marges ≤ 0)
    cur.execute(f"""
        SELECT COALESCE(SUM(
            CASE
              WHEN sold_price > 0 THEN
                CASE
                  WHEN margin IS NOT NULL THEN CASE WHEN margin > 0 THEN margin ELSE 0 END
                  ELSE CASE
                         WHEN (sold_price - COALESCE(cost_total, COALESCE(cost_accessory,0)+COALESCE(cost_fabrication,0))) > 0
                         THEN (sold_price - COALESCE(cost_total, COALESCE(cost_accessory,0)+COALESCE(cost_fabrication,0)))
                         ELSE 0
                       END
                END
              ELSE 0
            END
        ), 0)
        FROM objects
        {where}
    """, params)
    sum_positive_margin = float(cur.fetchone()[0] or 0.0)

    conn.close()
    return ObjectsSummary(
        total_objects=total_objects,
        sold_count=sold_count,
        available_count=available_count,
        gifted_count=gifted_count,
        personal_count=personal_count,
        sum_sold_price=sum_sold_price,
        sum_positive_margin=sum_positive_margin,
    )
    
def create_object_group(name: str) -> int:
    conn = _connect(); cur = conn.cursor()
    cur.execute("INSERT INTO object_groups(name) VALUES(?)", (name.strip()))
    gid = cur.lastrowid
    conn.commit(); conn.close()
    return gid

def rename_object_group(group_id: int, new_name: str) -> None:
    conn = _connect(); cur = conn.cursor()
    cur.execute("UPDATE object_groups SET name = ? WHERE id = ?", (new_name.strip(), group_id))
    conn.commit(); conn.close()

def assign_object_to_group(object_id: int, group_id: int) -> None:
    conn = _connect(); cur = conn.cursor()
    cur.execute("UPDATE objects SET object_group_id = ?, updated_at = datetime('now') WHERE id = ?", (group_id, object_id))
    if cur.rowcount == 0:
        conn.rollback(); conn.close()
        raise ValueError(f"Objet introuvable (id={object_id})")
    conn.commit(); conn.close()

def remove_object_from_group(object_id: int) -> None:
    conn = _connect(); cur = conn.cursor()
    cur.execute("UPDATE objects SET object_group_id = NULL, updated_at = datetime('now') WHERE id = ?", (object_id,))
    conn.commit(); conn.close()

def search_object_groups(query: str, limit: int = 10) -> list[dict]:
    conn = _connect(); cur = conn.cursor()
    q = f"%{(query or '').strip()}%"
    cur.execute("""
        SELECT id, name
          FROM object_groups
         WHERE name LIKE ?
         ORDER BY name ASC
         LIMIT ?
    """, (q, limit))
    rows = [{"id": r[0], "name": r[1]} for r in cur.fetchall()]
    conn.close()
    return rows

def list_object_groups_with_counts(filters: dict) -> list[dict]:
    """
    Retourne les groupes + compteurs + objets rattachés (filtrés comme la liste principale).
    - Comptes: available_count / total_count par groupe
    """
    where, params = _build_objects_where(filters)  # on réutilise tes filtres (search, sale_filter, etc.)

    conn = _connect(); cur = conn.cursor()

    # Total par groupe
    cur.execute(f"""
        SELECT og.id, og.name,
               COUNT(o.id)                           AS total_count,
               SUM(CASE WHEN o.available = 1 THEN 1 ELSE 0 END) AS available_count
          FROM object_groups og
     LEFT JOIN objects o ON o.object_group_id = og.id
         WHERE 1=1
      GROUP BY og.id
         HAVING total_count > 0
         ORDER BY og.name ASC
    """)
    groups = [{"id": gid, "name": name, "total": total, "available": (avail or 0)} for gid, name, total, avail in cur.fetchall()]

    # Objets par groupe, filtrés par les mêmes critères que la liste principale
    cur.execute(f"""
        SELECT o.*
          FROM objects o
         WHERE o.object_group_id IS NOT NULL
           AND o.id IN (SELECT id FROM objects {where})
         ORDER BY o.created_at DESC
    """, params)
    obj_by_group = {}
    cols = [d[0] for d in cur.description]
    for row in cur.fetchall():
        d = dict(zip(cols, row))
        obj_by_group.setdefault(d["object_group_id"], []).append(d)

    for g in groups:
        g["objects"] = obj_by_group.get(g["id"], [])

    conn.close()
    return groups


ensure_schema()