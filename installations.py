# installations.py
import os, json
from typing import List, Dict

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
INSTALLATIONS_FILE = os.path.join(DATA_DIR, 'installations.json')

os.makedirs(DATA_DIR, exist_ok=True)

def load_installations() -> List[Dict]:
    if os.path.exists(INSTALLATIONS_FILE):
        with open(INSTALLATIONS_FILE, 'r') as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []

def save_installations(installs: List[Dict]) -> None:
    with open(INSTALLATIONS_FILE, 'w') as f:
        json.dump(installs, f, indent=2)

def add_installation(label: str, guest_url: str) -> None:
    installs = load_installations()
    # id simple incrémental
    next_id = (max([i.get("id", 0) for i in installs]) + 1) if installs else 1
    installs.append({"id": next_id, "label": label.strip(), "guest_url": guest_url.strip()})
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
