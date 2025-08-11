# installations.py
import os, json
from typing import List, Dict
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
INSTALLATIONS_FILE = os.path.join(DATA_DIR, 'installations.json')
os.makedirs(DATA_DIR, exist_ok=True)

# remplace _normalize_guest_url par ceci
_DROP_KEYS = {'theme', 'origin', 'origin_label', 'current_label'}

def _normalize_guest_url(url: str) -> str:
    if not url:
        return url
    u = url.strip()
    p = urlparse(u)

    # Si l'URL a été saisie sans schéma (ex: example.com/guest/xxx)
    if not p.netloc:
        p = urlparse("https://" + u)

    # Force https si http ou vide
    scheme = 'https' if p.scheme in ('http', '') else p.scheme

    # Supprime les query params parasites
    q = [(k, v) for (k, v) in parse_qsl(p.query, keep_blank_values=True) if k not in _DROP_KEYS]

    # urlunparse attend 6 éléments: scheme, netloc, path, params, query, fragment
    return urlunparse((scheme, p.netloc, p.path, p.params, urlencode(q), p.fragment))

def load_installations() -> List[Dict]:
    if not os.path.exists(INSTALLATIONS_FILE):
        return []
    try:
        with open(INSTALLATIONS_FILE, 'r') as f:
            installs = json.load(f)
    except Exception:
        return []

    changed = False
    for it in installs:
        old = it.get("guest_url", "")
        new = _normalize_guest_url(old)
        if new != old:
            it["guest_url"] = new
            changed = True
    if changed:
        # auto-réécriture : on nettoie le fichier si on a patché quelque chose
        save_installations(installs)
    return installs

def save_installations(installs: List[Dict]) -> None:
    with open(INSTALLATIONS_FILE, 'w') as f:
        json.dump(installs, f, indent=2)

def add_installation(label: str, guest_url: str) -> None:
    installs = load_installations()
    next_id = (max([i.get("id", 0) for i in installs]) + 1) if installs else 1
    installs.append({
        "id": next_id,
        "label": (label or "").strip(),
        "guest_url": _normalize_guest_url(guest_url),
    })
    save_installations(installs)

def remove_installation(install_id: int) -> bool:
    installs = load_installations()
    new_list = [i for i in installs if i.get("id") != install_id]
    changed = len(new_list) != len(installs)
    if changed:
        save_installations(new_list)
    return changed

def get_installation(install_id: int) -> Dict:
    for i in load_installations():
        if i.get("id") == install_id:
            return i
    return {}
