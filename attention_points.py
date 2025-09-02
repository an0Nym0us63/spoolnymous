from __future__ import annotations
import random
import sqlite3
from typing import Any, Dict, List, Optional
import re

# Pas de paramètres en DB pour le moment : tout est dans ce fichier.
ATTENTION_SPOOL_EMPTY_THRESHOLD_G: float = 150.0   # seuil en grammes
ATTENTION_SPOOL_EMPTY_THRESHOLD_PCT: float = 0.15 # seuil en pourcentage (0.10 = 10%)

from filaments import fetch_spools
from print_history import db_config, list_print_images, list_group_images, get_print_groups
from datetime import datetime, timedelta

# Type uniforme pour un "point d'attention"
AttentionPoint = Dict[str, Any]

# ---------------------------------------------------------------------------
# Helpers DB & formatage
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(db_config["db_path"])
    conn.row_factory = sqlite3.Row
    return conn

_DISMISS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS attention_dismissed (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category TEXT NOT NULL,
  key_param TEXT NOT NULL,
  key_value TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NULL,
  UNIQUE(category, key_param, key_value)
);
"""

def _ensure_dismiss_table() -> None:
    conn = _get_conn()
    try:
        conn.execute(_DISMISS_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()

def _purge_expired_dismiss() -> None:
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM attention_dismissed WHERE expires_at IS NOT NULL AND expires_at < ?", (datetime.utcnow().isoformat(),))
        conn.commit()
    finally:
        conn.close()

def _point_identity(p: AttentionPoint) -> tuple[str, str, str]:
    """
    Retourne (category, key_param, key_value) pour identifier de façon stable un point.
    Convention : on utilise p['param'] et p['value'] s’ils existent, sinon on retombe
    sur ('name') histoire d’avoir une clé (moins robuste).
    """
    cat = str(p.get("category") or "")
    key_param = str(p.get("param") or "name")
    key_val = p.get("value")
    if key_val is None:
        key_val = p.get("name") or ""
    return cat, str(key_param), str(key_val)

def dismiss_point(category: str, key_param: str, key_value: str, *, ttl_days: Optional[int] = None) -> None:
    """
    Marque un point comme 'dismissed'. ttl_days=None => pas d’expiration.
    """
    _ensure_dismiss_table()
    _purge_expired_dismiss()
    now = datetime.utcnow()
    exp = (now + timedelta(days=int(ttl_days))) if isinstance(ttl_days, int) and ttl_days > 0 else None

    conn = _get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO attention_dismissed (category, key_param, key_value, created_at, expires_at) VALUES (?,?,?,?,?)",
            (str(category), str(key_param), str(key_value), now.isoformat(), exp.isoformat() if exp else None),
        )
        conn.commit()
    finally:
        conn.close()

def restore_point(category: str, key_param: str, key_value: str) -> None:
    """Annule le dismiss d’un point."""
    _ensure_dismiss_table()
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM attention_dismissed WHERE category=? AND key_param=? AND key_value=?",
                     (str(category), str(key_param), str(key_value)))
        conn.commit()
    finally:
        conn.close()

def _load_dismissed_index() -> set[tuple[str, str, str]]:
    """
    Retourne un set de triples (category, key_param, key_value) actifs.
    """
    _ensure_dismiss_table()
    _purge_expired_dismiss()
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT category, key_param, key_value FROM attention_dismissed")
        rows = cur.fetchall()
        return {(str(r["category"]), str(r["key_param"]), str(r["key_value"])) for r in rows}
    finally:
        conn.close()

def _filter_buckets_dismissed(buckets: Dict[str, List[AttentionPoint]]) -> Dict[str, List[AttentionPoint]]:
    """
    Filtre les points qui ont été dismiss (toutes catégories).
    """
    dismissed = _load_dismissed_index()
    out: Dict[str, List[AttentionPoint]] = {}
    for cat, items in (buckets or {}).items():
        kept: List[AttentionPoint] = []
        for p in items or []:
            k = _point_identity(p)
            if k in dismissed:
                continue
            kept.append(p)
        out[cat] = kept
    return out

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


def _normalize_hex(color: Optional[str]) -> Optional[str]:
    """Retourne une couleur hex valide (#RRGGBB) si possible, sinon None."""
    if not color or not isinstance(color, str):
        return None
    c = color.strip()
    if not c:
        return None
    if c.startswith('#'):
        c = c.upper()
        if len(c) == 7 and all(ch in '0123456789ABCDEF#' for ch in c):
            return c
        if len(c) == 4 and all(ch in '0123456789ABCDEF#' for ch in c):
            # convert #RGB -> #RRGGBB
            return f"#{c[1]*2}{c[2]*2}{c[3]*2}"
    return None


def _parse_color_field(raw: Optional[Any]) -> Dict[str, Any]:
    """Tente d'interpréter le champ couleur en :
    - color_hex : str (#RRGGBB) si une seule couleur
    - colors    : List[str] si plusieurs hex (pour gradient)
    - color_name: str si c'est un nom non-hex
    Accepte soit une string ("#ff0000" ou "#ff0000,#00ff00") soit une liste.
    """
    meta: Dict[str, Any] = {"color_hex": None, "colors": None, "color_name": None}
    if raw is None:
        return meta
    # liste -> normaliser chaque entrée hex
    if isinstance(raw, (list, tuple)):
        colors = [c for c in ( _normalize_hex(str(x)) for x in raw ) if c]
        if len(colors) == 1:
            meta["color_hex"] = colors[0]
        elif len(colors) > 1:
            meta["colors"] = colors
        return meta
    # string -> split sur , ; |
    if isinstance(raw, str):
        # essais multi
        if any(sep in raw for sep in [',',';','|']):
            parts = [p.strip() for p in re.split(r"[,;|]", raw) if p.strip()]
            colors = [c for c in (_normalize_hex(p) for p in parts) if c]
            if len(colors) == 1:
                meta["color_hex"] = colors[0]
            elif len(colors) > 1:
                meta["colors"] = colors
            else:
                meta["color_name"] = raw.strip()
            return meta
        # simple
        hx = _normalize_hex(raw)
        if hx:
            meta["color_hex"] = hx
        else:
            meta["color_name"] = raw.strip()
        return meta
    # fallback texte
    meta["color_name"] = str(raw)
    return meta


def _swatch_html_from_meta(meta: Dict[str, Any], size: int = 12, radius: int = 3) -> str:
    """Construit une vignette HTML inline (span) :
    - couleur unie si `color_hex`
    - gradient linéaire si `colors`
    Retourne une string HTML prête à insérer dans une phrase.
    """
    style_base = f"display:inline-block;width:{size}px;height:{size}px;border-radius:{radius}px;vertical-align:middle;margin:0 6px 0 2px;border:1px solid rgba(0,0,0,.15);"
    if meta.get("colors"):
        grad = ", ".join(meta["colors"])  # déjà normalisées
        style = style_base + f"background: linear-gradient(90deg, {grad});"
        return f"<span aria-hidden=\"true\" style=\"{style}\"></span>"
    if meta.get("color_hex"):
        style = style_base + f"background-color: {meta['color_hex']};"
        return f"<span aria-hidden=\"true\" style=\"{style}\"></span>"
    return ""
    c = color.strip()
    if not c:
        return None
    if c.startswith('#'):
        c = c.upper()
        if len(c) == 7 and all(ch in '0123456789ABCDEF#' for ch in c):
            return c
        if len(c) == 4 and all(ch in '0123456789ABCDEF#' for ch in c):
            # convert #RGB -> #RRGGBB
            return f"#{c[1]*2}{c[2]*2}{c[3]*2}"
    # sinon, on ne tente pas de conversion nom->hex ici
    return None

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
    cur.execute(
        """
        SELECT f.id, f.name, f.manufacturer, f.material, f.color
        FROM filaments f
        WHERE COALESCE(f.swatch, 0) = 0
        ORDER BY f.manufacturer, f.name
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    out: List[AttentionPoint] = []
    for r in rows:
        name_parts = [p for p in [r.get("manufacturer"), r.get("name"), r.get("material")] if p]
        disp = " — ".join(name_parts) if name_parts else f"Filament #{r['id']}"
        color_meta = _parse_color_field(r.get("color"))
        meta = {**color_meta}
        out.append({
            "category": "filament_without_swatch",
            "name": disp,
            "param": "filament_id",
            "value": int(r["id"]),
            "meta": meta,
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
   
def _collect_prints_photo_without_design(limit: int = 500) -> List[AttentionPoint]:
    """
    Sélectionne les impressions qui ont au moins une photo uploadée manuellement
    (via list_print_images) mais dont design_id est vide / nul / 0.
    """
    conn = _get_conn()
    cur = conn.cursor()
    # On couvre: NULL, 0, '', '  '
    cur.execute(f"""
        SELECT p.id, p.file_name, p.design_id
        FROM prints p
        WHERE p.design_id IS NULL
              OR p.design_id = 0
              OR TRIM(CAST(p.design_id AS TEXT)) = ''
        ORDER BY p.id DESC
        LIMIT {int(limit)}
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    out: List[AttentionPoint] = []
    for r in rows:
        imgs = list_print_images(r["id"])
        if imgs:  # => il y a bien une (vraie) photo côté uploads
            out.append({
                "category": "print_photo_without_design",
                "name": r["file_name"],
                "param": "print_id",
                "value": int(r["id"]),
                "meta": {
                    "images_count": len(imgs),
                    "design_id": r.get("design_id"),
                }
            })
    return out

# ---------------------------------------------------------------------------
# Rendu des messages
# ---------------------------------------------------------------------------

PHRASES = {
    "spool_almost_empty": [
        "{name} est presque vide ({remaining_g:.0f} g restants{pct_txt}). Essayez de planifier une petite impression pour la terminer.",
        "{name} arrive en fin de bobine ({remaining_g:.0f} g{pct_txt}). C’est le moment de l’utiliser sur une petite pièce.",
        "Attention, {name} est faible ({remaining_g:.0f} g{pct_txt}).",
        "{name} est presque au bout ({remaining_g:.0f} g{pct_txt}).",
        "Stock faible sur {name} : {remaining_g:.0f} g{pct_txt} restants.",
        "{name} pourrait ne pas suffire pour une grosse pièce ({remaining_g:.0f} g{pct_txt}).",
        "Derniers grammes pour {name} : {remaining_g:.0f} g{pct_txt}.",
    ],
    "filament_without_swatch": [
        "Filament sans swatch {swatch_html}: {name}. Ajoutez un échantillon.",
        "Pas de swatch {swatch_html} pour {name}. Une vignette couleur serait utile.",
        "{swatch_html} {name} n’a pas encore de swatch. Pensez à le créer.",
        "Swatch manquant {swatch_html}: {name}.",
        "Échantillon absent {swatch_html} pour {name}.",
        "Complétez le swatch de {swatch_html} {name} pour le catalogue.",
        "{name} : aucun swatch enregistré {swatch_html}.",
        "Ajoutez une pastille {swatch_html} pour {name}.",
        "{name} est sans aperçu couleur (swatch) {swatch_html}.",
        "Faites une prise rapide : swatch manquant sur {name} {swatch_html}.",
    ],
    "print_usage_unassigned": [
        "Affectations manquantes : l’impression « {name} » a {usages} usage(s) non associés (≈ {grams_total:.0f} g).",
        "L’impression « {name} » a des usages de filament non reliés à une bobine (≈ {grams_total:.0f} g).",
        "Associez les usages de filament pour « {name} » ({usages} entrées, ~{grams_total:.0f} g).",
        "Usages orphelins sur « {name} » : {usages} (≈ {grams_total:.0f} g).",
        "Lien bobine manquant pour « {name} » (~{grams_total:.0f} g).",
        "Vérifier l’affectation de filament pour « {name} ».",
        "« {name} » : usages sans bobine ({usages}).",
        "Affectez une bobine aux usages de « {name} ».",
        "Usages non assignés (≈ {grams_total:.0f} g) sur « {name} ».",
        "Compléter les affectations pour « {name} ».",
    ],
    "print_without_photo": [
        "Photo manquante : l’impression « {name} » n’a pas de visuel.",
        "Ajoutez une image pour l’impression « {name} ».",
        "Pas d’aperçu pour « {name} ». Pensez à une photo.",
        "Aucun visuel enregistré pour « {name} ».",
        "« {name} » mérite une photo !",
        "Capturez un cliché de « {name} » pour l’historique.",
        "Complétez la galerie de « {name} ».",
        "Visuel absent sur « {name} ».",
        "Ajoutez une miniature pour « {name} ».",
        "Photo à ajouter pour « {name} ».",
    ],
    "group_without_photo": [
        "Photo manquante : le groupe « {name} » n’a pas de visuel.",
        "Ajoutez une image au groupe « {name} ».",
        "Pas d’aperçu pour le groupe « {name} ».",
        "Aucun visuel enregistré pour le groupe « {name} ».",
        "« {name} » (groupe) mérite une photo.",
        "Complétez la galerie du groupe « {name} ».",
        "Visuel absent sur le groupe « {name} ».",
        "Ajoutez une miniature pour le groupe « {name} ».",
        "Photo à ajouter pour le groupe « {name} ».",
        "Pensez à illustrer le groupe « {name} ».",
    ],
    "print_photo_without_design": [
        "Photo présente mais design non lié : « {name} » ({images_count} image(s)). Associez un design.",
        "« {name} » a une photo mais aucun design associé. Renseignez le design_id.",
        "Design manquant pour « {name} » alors qu’une photo existe ({images_count}).",
        "Liez un design à « {name} » : photo déjà uploadée.",
        "« {name} » : visuel OK, design_id absent.",
    ],
}

def render_message(point: AttentionPoint) -> str:
    cat = point.get("category")
    tpls = PHRASES.get(cat, ["{name}"])
    tpl = random.choice(tpls) if isinstance(tpls, list) and tpls else tpls
    meta = point.get("meta") or {}

    # Ajout contextuel : vignette/gradient au sein de la phrase pour les filaments sans swatch
    if cat == "filament_without_swatch":
        swatch_html = _swatch_html_from_meta(meta)
    else:
        swatch_html = ""

    d = {
        "name": point.get("name", ""),
        "swatch_html": swatch_html,
        **meta,
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
        "print_photo_without_design": _collect_prints_photo_without_design(),
    }

def sample_for_home(per_category_max: int = 3, *, exclude_dismissed: bool = True) -> List[AttentionPoint]:
    """
    Récupère collect_attention_points(), filtre éventuellement les points dismiss,
    puis sélectionne aléatoirement jusqu’à N points par catégorie.
    """
    buckets = collect_attention_points()

    # Option : filtrer ici
    if exclude_dismissed:
        dismissed = _load_dismissed_index()  # set[(category, key_param, key_value)]
        def kept(points: List[AttentionPoint]) -> List[AttentionPoint]:
            out: List[AttentionPoint] = []
            for p in points:
                cat, key_param, key_val = _point_identity(p)
                if (cat, key_param, key_val) in dismissed:
                    continue
                out.append(p)
            return out
        buckets = {cat: kept(items or []) for cat, items in buckets.items()}

    # Échantillonnage par catégorie
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
    "print_photo_without_design": "Photos sans design",
}


def build_buckets_with_messages(
    buckets: Dict[str, List[AttentionPoint]],
    *,
    sample_per_category: Optional[int] = None,
) -> Dict[str, List[AttentionPoint]]:
    """Ajoute la clé `message` à chaque point et **peut échantillonner** par catégorie.

    Args:
        buckets: dict {cat: [points...]}
        sample_per_category: si défini, limite aléatoirement à N éléments par catégorie
                             (None pour conserver la liste complète)
    """
    out: Dict[str, List[AttentionPoint]] = {}
    for cat, items in buckets.items():
        chosen = items
        if items and isinstance(sample_per_category, int):
            n = min(len(items), max(sample_per_category, 0))
            chosen = random.sample(items, n)
        out[cat] = [dict(p, message=render_message(p)) for p in chosen]
    return out


def get_attention_context(per_category_max: int = 3, *, sample_buckets: bool = True) -> Dict[str, Any]:
    """
    Prépare tout ce qu'il faut passer au template.

    Args:
        per_category_max: nombre max d'items par catégorie pour l'accueil
        sample_buckets: si True, **échantillonne** aussi les listes par catégorie; sinon, renvoie toutes les entrées
    """
    buckets = collect_attention_points()
    filtered_buckets = _filter_buckets_dismissed(buckets)
    buckets_rendered = build_buckets_with_messages(
        filtered_buckets,
        sample_per_category=per_category_max if sample_buckets else None,
    )

    samples = sample_for_home(per_category_max=per_category_max)
    samples = [dict(p, message=render_message(p)) for p in samples]

    return {
        "attention_samples": samples,
        "attention_buckets": buckets_rendered,
        "category_labels": CATEGORY_LABELS,
    }
