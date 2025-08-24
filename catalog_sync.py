# services/catalog_sync.py
from __future__ import annotations
import os, json, time, hashlib, threading, logging
from typing import Optional, Tuple
import requests

log = logging.getLogger(__name__)

# --- Config en dur (paramétrables ici) ---
DATA_DIR = os.path.join("data", "filaments")
INTERVAL_SEC = 60
TIMEOUT_SEC = 15
USER_AGENT = "SpoolNymous-CatalogSync/1.0"
FILAMENTS_URL = "https://donkie.github.io/SpoolmanDB/filaments.json"
MATERIALS_URL = "https://donkie.github.io/SpoolmanDB/materials.json"
FILAMENTS_FILE = "filaments.json"
MATERIALS_FILE = "materials.json"
MANIFEST_FILE = "manifest.json"
# ----------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _write_if_changed(path: str, content_bytes: bytes) -> bool:
    new_hash = _sha256_bytes(content_bytes)
    meta_path = path + ".sha256"
    old_hash = None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            old_hash = f.read().strip()
    except FileNotFoundError:
        pass
    if new_hash != old_hash:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(content_bytes)
        os.replace(tmp, path)
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(new_hash)
        return True
    return False

def _etag_path(target_path: str) -> str:
    return target_path + ".etag"

def _load_etag(target_path: str) -> Optional[str]:
    try:
        with open(_etag_path(target_path), "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except FileNotFoundError:
        return None

def _save_etag(target_path: str, etag: Optional[str]) -> None:
    if etag is None:
        try: os.remove(_etag_path(target_path))
        except FileNotFoundError: pass
        return
    with open(_etag_path(target_path), "w", encoding="utf-8") as f:
        f.write(etag)

class CatalogSync:
    """
    - sync_once() : télécharge filaments/materials (+ manifest) dans data/filaments/
    - start() : lance une boucle périodique (INTERVAL_SEC)
    - stop()  : arrête proprement la boucle
    """
    def __init__(self,
                 data_dir: str = DATA_DIR,
                 interval_sec: int = INTERVAL_SEC,
                 timeout_sec: int = TIMEOUT_SEC) -> None:
        self.data_dir = data_dir
        self.interval_sec = interval_sec
        self.timeout_sec = timeout_sec
        self._stop_evt = threading.Event()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})
        _ensure_dir(self.data_dir)

    def _fetch_to_file(self, url: str, filename: str) -> Tuple[bool, Optional[int]]:
        target = os.path.join(self.data_dir, filename)
        etag = _load_etag(target)
        headers = {"Accept": "application/json"}
        if etag:
            headers["If-None-Match"] = etag
        try:
            resp = self._session.get(url, headers=headers, timeout=self.timeout_sec)
        except Exception as e:
            log.exception("Erreur réseau %s: %s", url, e)
            return (False, None)

        if resp.status_code == 304:
            log.info("%s inchangé (304)", filename)
            return (False, 304)
        if resp.status_code != 200:
            log.warning("HTTP %s pour %s", resp.status_code, url)
            return (False, resp.status_code)

        try:
            data = resp.json()  # validation
            pretty = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        except Exception as e:
            log.warning("Contenu non-JSON pour %s (%s), ignoré.", url, e)
            return (False, resp.status_code)

        written = _write_if_changed(target, pretty)
        if written:
            log.info("Écrit %s (%d octets).", filename, len(pretty))
        else:
            log.info("%s sans changement (hash identique).", filename)

        _save_etag(target, resp.headers.get("ETag"))
        return (written, resp.status_code)

    def sync_once(self) -> dict:
        started = time.time()
        w_fil, st_fil = self._fetch_to_file(FILAMENTS_URL, FILAMENTS_FILE)
        w_mat, st_mat = self._fetch_to_file(MATERIALS_URL, MATERIALS_FILE)

        manifest = {
            "synced_at": int(time.time()),
            "files": [FILAMENTS_FILE, MATERIALS_FILE],
            "sources": {FILAMENTS_FILE: FILAMENTS_URL, MATERIALS_FILE: MATERIALS_URL},
            "last_status": {FILAMENTS_FILE: st_fil, MATERIALS_FILE: st_mat},
            "written": {"filaments": bool(w_fil), "materials": bool(w_mat)},
        }
        _write_if_changed(os.path.join(self.data_dir, MANIFEST_FILE),
                          json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
        took = time.time() - started
        log.info("Sync catalog terminée en %.2fs (écrit: filaments=%s, materials=%s)", took, w_fil, w_mat)
        return manifest

    def run_periodic(self) -> None:
        log.info("Démarrage CatalogSync (dir=%s, interval=%ss)", self.data_dir, self.interval_sec)
        try:
            self.sync_once()
        except Exception:
            log.exception("Sync initiale échouée")
        while not self._stop_evt.wait(self.interval_sec):
            try:
                self.sync_once()
            except Exception:
                log.exception("Sync périodique échouée")
        log.info("Arrêt CatalogSync.")

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run_periodic, name="CatalogSync", daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop_evt.set()
