from __future__ import annotations
import random
import sqlite3
from typing import Any, Dict, List, Optional

# Pas de paramètres en DB pour le moment : tout est dans ce fichier.
ATTENTION_SPOOL_EMPTY_THRESHOLD_G: float = 50.0   # seuil en grammes
ATTENTION_SPOOL_EMPTY_THRESHOLD_PCT: float = 0.10 # seuil en pourcentage (0.10 = 10%)

from filaments import fetch_spools
from print_history import db_config, list_print_images, list_group_images, get_print_groups

# Type uniforme pour un "point d'attention"
AttentionPoint = Dict[str, Any]

# ---------------------------------------------------------------------------
# Helpers DB & formatage
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    return conn

def _row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}

def _fmt_spool_name(spool: dict) -> str:
    f = spool.get("filament", {}) or {}
    man = f.get("manufacturer") or ""
    name = f.get("name") or ""
    color = f.get("color_hex") or f.get("color") or ""
    sid = spool.get("id") or spool.get("spool_id") or ""
    parts = []
    if man: parts.append(man)
    if name: parts.append(name)
    lib = " — ".join(parts) if parts else f"Spool #{sid}"
    if color: lib = f"{lib} ({color})"
    if sid: lib = f"{lib}  #{sid}"
    return lib.strip()

# ---------------------------------------------------------------------------
# Collecteurs par catégorie
# ---------------------------------------------------------------------------

def _collect_unassigned_filament_usage() -> List[AttentionPoint]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT p.id AS print_id, p.file_name, p.print_date,
               SUM(fu.grams_used) AS grams_total,
               COUNT(*) AS usages
        FROM filament_usage fu
        JOIN prints p ON p.id = fu.print_id
        WHERE fu.spool_id IS NULL
        GROUP BY p.id, p.file_name, p.print_date
        ORDER BY p.print_date DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    out: List[AttentionPoint] = []
    for r in rows:
        out.append({
            "category": "print_usage_unassigned",
            "name": r["file_name"],
            "param": "print_id",
            "value": int(r["print_id"]),
            "meta": {
                "grams_total": float(r["grams_total"] or 0.0),
                "usages": int(r["usages"] or 0),
                "print_date": r["print_date"],
            }
        })
    return out

def _collect_filaments_without_swatch() -> List[AttentionPoint]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT f.id, f.name, f.manufacturer, f.material, f.color
        FROM filaments f
        WHERE COALESCE(f.swatch, 0) = 0
        ORDER BY f.manufacturer, f.name
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    out: List[AttentionPoint] = []
    for r in rows:
        name_parts = [p for p in [r.get("manufacturer"), r.get("name"), r.get("material")] if p]
        disp = " — ".join(name_parts) if name_parts else f"Filament #{r['id']}"
        color = r.get("color")
        if color:
            disp = f"{disp} ({color})"
        out.append({
            "category": "filament_without_swatch",
            "name": disp,
            "param": "filament_id",
            "value": int(r["id"]),
            "meta": {}
        })
    return out

def _collect_spools_almost_empty() -> List[AttentionPoint]:
    th_g = float(ATTENTION_SPOOL_EMPTY_THRESHOLD_G)
    th_pct = float(ATTENTION_SPOOL_EMPTY_THRESHOLD_PCT)

    spools = fetch_spools(archived=False)
    out: List[AttentionPoint] = []
    for s in spools:
        rem_g = s.get("remaining_weight")
        fil = s.get("filament") or {}
        total_g = fil.get("weight")
        pct: Optional[float] = None
        if rem_g is not None and total_g:
            try:
                pct = float(rem_g) / float(total_g)
            except Exception:
                pct = None

        is_low_by_g = (rem_g is not None and float(rem_g) <= th_g)
        is_low_by_pct = (pct is not None and pct <= th_pct)

        if is_low_by_g or is_low_by_pct:
            out.append({
                "category": "spool_almost_empty",
                "name": _fmt_spool_name(s),
                "param": "spool_id",
                "value": int(s.get("id") or s.get("spool_id") or 0),
                "meta": {
                    "remaining_g": float(rem_g) if rem_g is not None else None,
                    "pct": float(pct) if pct is not None else None,
                    "total_g": float(total_g) if total_g is not None else None,
                    "threshold_g": th_g,
                    "threshold_pct": th_pct,
                }
            })
    return out

def _collect_prints_without_photo(limit: int = 500) -> List[AttentionPoint]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(f"""
        SELECT p.id, p.file_name
        FROM prints p
        ORDER BY p.id DESC
        LIMIT {int(limit)}
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    out: List[AttentionPoint] = []
    for r in rows:
        imgs = list_print_images(r["id"])
        if not imgs:
            out.append({
                "category": "print_without_photo",
                "name": r["file_name"],
                "param": "print_id",
                "value": int(r["id"]),
                "meta": {}
            })
    return out

def _collect_groups_without_photo() -> List[AttentionPoint]:
    groups = get_print_groups()
    out: List[AttentionPoint] = []
    for g in groups:
        imgs = list_group_images(g["id"])
        if not imgs:
            out.append({
                "category": "group_without_photo",
                "name": g.get("name") or f"Groupe #{g['id']}",
                "param": "group_id",
                "value": int(g["id"]),
                "meta": {}
            })
    return out

# ---------------------------------------------------------------------------
# Rendu des messages
# ---------------------------------------------------------------------------

PHRASES = {
    "spool_almost_empty": (
        "Bobine à finir : {name} est presque vide ({remaining_g:.0f} g restants{pct_txt}). "
        "Essayez de planifier une petite impression pour la terminer."
    ),
    "filament_without_swatch": (
        "Filament sans swatch : {name}. Prenez une photo/échantillon pour améliorer le catalogue."
    ),
    "print_usage_unassigned": (
        "Affectations manquantes : l’impression « {name} » a {usages} usage(s) de filament "
        "non associés à une bobine (≈ {grams_total:.0f} g)."
    ),
    "print_without_photo": (
        "Photo manquante : l’impression « {name} » n’a pas encore de visuel."
    ),
    "group_without_photo": (
        "Photo manquante : le groupe « {name} » n’a pas encore de visuel."
    ),
}

def render_message(point: AttentionPoint) -> str:
    cat = point.get("category")
    tpl = PHRASES.get(cat, "{name}")
    meta = point.get("meta") or {}
    d = {
        "name": point.get("name", ""),
        **meta
    }
    if cat == "spool_almost_empty":
        pct = meta.get("pct")
        d["pct_txt"] = f", {pct*100:.0f}%" if isinstance(pct, (int, float)) else ""
        if d.get("remaining_g") is None:
            d["remaining_g"] = 0.0
    return tpl.format(**d)

# ---------------------------------------------------------------------------
# Orchestrateur
# ---------------------------------------------------------------------------

def collect_attention_points() -> Dict[str, List[AttentionPoint]]:
    return {
        "print_usage_unassigned": _collect_unassigned_filament_usage(),
        "filament_without_swatch": _collect_filaments_without_swatch(),
        "spool_almost_empty": _collect_spools_almost_empty(),
        "print_without_photo": _collect_prints_without_photo(),
        "group_without_photo": _collect_groups_without_photo(),
    }

def sample_for_home(per_category_max: int = 3) -> List[AttentionPoint]:
    """
    Récupère directement les points via collect_attention_points et sélectionne
    aléatoirement jusqu’à N points par catégorie.
    """
    buckets = collect_attention_points()
    out: List[AttentionPoint] = []
    for cat, items in buckets.items():
        if not items:
            continue
        n = min(len(items), per_category_max)
        out.extend(random.sample(items, n))
    random.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Contexte prêt pour le template (labels + messages + échantillon)
# ---------------------------------------------------------------------------

CATEGORY_LABELS: Dict[str, str] = {
    "print_usage_unassigned": "Affectations manquantes",
    "filament_without_swatch": "Filaments sans swatch",
    "spool_almost_empty": "Bobines presque vides",
    "print_without_photo": "Impressions sans photo",
    "group_without_photo": "Groupes sans photo",
}


def build_buckets_with_messages(buckets: Dict[str, List[AttentionPoint]]) -> Dict[str, List[AttentionPoint]]:
    """Ajoute la clé `message` à chaque point en utilisant `render_message`."""
    out: Dict[str, List[AttentionPoint]] = {}
    for cat, items in buckets.items():
        out[cat] = [dict(p, message=render_message(p)) for p in items]
    return out


def get_attention_context(per_category_max: int = 3) -> Dict[str, Any]:
    """
    Prépare tout ce qu'il faut passer au template, sans que `app.py` n'ait à
    reconstruire quoi que ce soit.

    Retour :
      - attention_samples : liste de points (avec `message`) issus du sampling
      - attention_buckets : dict catégorisé -> liste de points (avec `message`)
      - category_labels   : mapping clé catégorie -> libellé lisible
    """
    buckets = collect_attention_points()
    buckets_rendered = build_buckets_with_messages(buckets)

    samples = sample_for_home(per_category_max=per_category_max)
    samples = [dict(p, message=render_message(p)) for p in samples]

    return {
        "attention_samples": samples,
        "attention_buckets": buckets_rendered,
        "category_labels": CATEGORY_LABELS,
    }
