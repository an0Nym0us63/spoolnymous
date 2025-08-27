import json
import ssl
import traceback
from threading import Thread, Lock

from datetime import datetime, timedelta
import paho.mqtt.client as mqtt

from config import get_app_setting, AUTO_SPEND, EXTERNAL_SPOOL_AMS_ID, EXTERNAL_SPOOL_ID
from messages import GET_VERSION, PUSH_ALL
from tools_3mf import getMetaDataFrom3mf
import time
import copy
import math
from typing import Dict, Any
from collections.abc import Mapping
from logger import append_to_rotating_file
from print_history import  insert_print, insert_filament_usage, update_filament_spool,update_print_field_with_job_id,get_tray_spool_map,delete_tray_spool_map_by_id,snapshot_milestone
from filaments import fetch_spools,clearActiveTray,setActiveTray,spendFilaments
from globals import PRINTER_STATUS, PRINTER_STATUS_LOCK, PROCESSED_JOBS, PENDING_JOBS
import logging
from pathlib import Path
from contextlib import suppress

logger = logging.getLogger(__name__)



# État mémoire par job_id (évite doublons durant le run)
MQTT_CLIENT = {}  # Global variable storing MQTT Client
MQTT_CLIENT_CONNECTED = False
MQTT_KEEPALIVE = 60
LAST_AMS_CONFIG = {}  # Global variable storing last AMS configuration

PRINTER_STATE = {}
PRINTER_STATE_LAST = {}

PENDING_PRINT_METADATA = {}

# --- Async helper pour ne pas bloquer le thread MQTT ---
PROCESSMSG_LOCK = Lock()  # empêche les exécutions concurrentes de processMessage


def _milestones_to_fire(prev: float, curr: float, done_50: bool, done_99: bool, done_100: bool):
    """
    Retourne la liste des milestones à déclencher maintenant, parmi [50, 99, 100].
    Règles:
      - une seule fois par palier,
      - rattrapage si on n’a pas vu la valeur exacte (déclenche au 1er passage >= palier),
      - l’ordre n’est pas garanti (on peut recevoir 100 directement).
    """
    fire = []
    if not done_50 and prev < 50.0 and curr >= 50.0:
        fire.append(50)
    if not done_99 and prev < 99.0 and curr >= 99.0:
        fire.append(99)
    if not done_100 and prev < 100.0 and curr >= 100.0:
        fire.append(100)
    return fire

def fire_and_forget(fn, *args, name=None, release_lock=False, **kwargs):
    """
    Lance une fonction en thread séparé, sans bloquer.
    - log start/done/crash
    - option release_lock=True pour libérer PROCESSMSG_LOCK en fin de tâche
    """
    def _runner():
        t0 = time.time()
        try:
            logger.info(f"[async] {fn.__name__} start")
            fn(*args, **kwargs)
            dt = time.time() - t0
            logger.info(f"[async] {fn.__name__} done in {dt:.1f}s")
        except Exception:
            logger.exception(f"[async] {fn.__name__} crashed")
        finally:
            if release_lock:
                try:
                    PROCESSMSG_LOCK.release()
                except RuntimeError:
                    pass

    t = Thread(target=_runner, name=name or fn.__name__, daemon=True)
    t.start()
    return t

def update_status(new_data):
    with PRINTER_STATUS_LOCK:
        PRINTER_STATUS.update(new_data)
        
def getPrinterModel():
    PRINTER_ID = get_app_setting("PRINTER_ID","")
    model_code = PRINTER_ID[:3]

    model_map = {
        "094": "H2D",
        "00W": "X1",
        "00M": "X1 Carbon",
        "03W": "X1E",
        "01S": "P1P",
        "01P": "P1S",
        "039": "A1",
        "030": "A1 Mini"
    }
    model_name = model_map.get(model_code, f"Unknown model ({model_code})")

    numeric_tail = ''.join(filter(str.isdigit, PRINTER_ID))
    device_id = numeric_tail[-3:] if len(numeric_tail) >= 3 else numeric_tail

    device_name = f"3DP-{model_code}-{device_id}"

    return {
        "model": model_name,
        "devicename": device_name
    }

def num2letter(num):
  return chr(ord("A") + int(num))
  
def update_dict(original: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, Mapping) and key in original and isinstance(original[key], Mapping):
            original[key] = update_dict(original[key], value)
        else:
            original[key] = value
    return original

def map_filament(tray_tar):
  global PENDING_PRINT_METADATA
  # Prüfen, ob ein Filamentwechsel aktiv ist (stg_cur == 4)
  #if stg_cur == 4 and tray_tar is not None:
  if PENDING_PRINT_METADATA:
    PENDING_PRINT_METADATA["filamentChanges"].append(tray_tar)  # Jeder Wechsel zählt, auch auf das gleiche Tray
    logger.info(f'Filamentchange {len(PENDING_PRINT_METADATA["filamentChanges"])}: Tray {tray_tar}')

    # Anzahl der erkannten Wechsel
    change_count = len(PENDING_PRINT_METADATA["filamentChanges"]) - 1  # -1, weil der erste Eintrag kein Wechsel ist

    # Slot in der Wechselreihenfolge bestimmen
    for tray, usage_count in PENDING_PRINT_METADATA["filamentOrder"].items():
        if usage_count == change_count:
            PENDING_PRINT_METADATA["ams_mapping"].append(tray_tar)
            logger.info(f"✅ Tray {tray_tar} assigned Filament to {tray}")

            for filament, tray in enumerate(PENDING_PRINT_METADATA["ams_mapping"]):
              logger.info(f"  Filament {filament} → Tray {tray}")


    # Falls alle Slots zugeordnet sind, Ausgabe der Zuordnung
    if len(PENDING_PRINT_METADATA["ams_mapping"]) == len(PENDING_PRINT_METADATA["filamentOrder"]):
        logger.info("\n✅ All trays assigned:")
        for filament, tray in enumerate(PENDING_PRINT_METADATA["ams_mapping"]):
            logger.info(f"  Filament {tray} → Tray {tray}")

        return True
  
  return False
  
def processMessage(data):
  global LAST_AMS_CONFIG, PRINTER_STATE, PRINTER_STATE_LAST, PENDING_PRINT_METADATA
    
   # Prepare AMS spending estimation
  if "print" in data:
    #logger.info(str(data))
    if "command" in data["print"] and data["print"]["command"] == "project_file" and "url" in data["print"]:
      logger.info('1'+str(data))
      PENDING_PRINT_METADATA = getMetaDataFrom3mf(data["print"]["url"],data["print"]["subtask_name"])
      name=PRINTER_STATE["print"]["subtask_name"]
      if PENDING_PRINT_METADATA["title"] != '':
        name = PENDING_PRINT_METADATA["title"]
      if (PENDING_PRINT_METADATA["plateID"] != '1'):
        name += ' - ' +PENDING_PRINT_METADATA["plateID"]
      logger.info('Inserting new print ' + name)
      logger.debug(str(PENDING_PRINT_METADATA))
      print_id = insert_print(name, "cloud", PENDING_PRINT_METADATA["image"],None,PENDING_PRINT_METADATA["duration"],data["print"]["job_id"])

      if "use_ams" in PRINTER_STATE["print"] and PRINTER_STATE["print"]["use_ams"]:
        PENDING_PRINT_METADATA["ams_mapping"] = PRINTER_STATE["print"]["ams_mapping"]
      else:
        PENDING_PRINT_METADATA["ams_mapping"] = [EXTERNAL_SPOOL_ID]

      PENDING_PRINT_METADATA["print_id"] = print_id
      PENDING_PRINT_METADATA["complete"] = True

      for id, filament in PENDING_PRINT_METADATA["filaments"].items():
        insert_filament_usage(print_id, filament["type"], filament["color"], filament["used_g"], id)
  
    #if ("gcode_state" in data["print"] and data["print"]["gcode_state"] == "RUNNING") and ("print_type" in data["print"] and data["print"]["print_type"] != "local") \
    #  and ("tray_tar" in data["print"] and data["print"]["tray_tar"] != "255") and ("stg_cur" in data["print"] and data["print"]["stg_cur"] == 0 and PRINT_CURRENT_STAGE != 0):
    
    #TODO: What happens when printed from external spool, is ams and tray_tar set?
    if ( "print_type" in PRINTER_STATE["print"] and PRINTER_STATE["print"]["print_type"] == "local" and
        "print" in PRINTER_STATE_LAST
      ):
      logger.info('2'+str(data))
      if (
          "gcode_state" in PRINTER_STATE["print"] and 
          PRINTER_STATE["print"]["gcode_state"] == "RUNNING" and
          PRINTER_STATE_LAST["print"]["gcode_state"] == "PREPARE" and 
          "gcode_file" in PRINTER_STATE["print"]
        ):

        PENDING_PRINT_METADATA = getMetaDataFrom3mf(PRINTER_STATE["print"]["gcode_file"])
        name=PENDING_PRINT_METADATA["file"]
        if PENDING_PRINT_METADATA["title"] != '':
            name = PENDING_PRINT_METADATA["title"]
        if (PENDING_PRINT_METADATA["plateID"] != '1'):
            name += ' - ' +PENDING_PRINT_METADATA["plateID"]
        print_id = insert_print(name, PRINTER_STATE["print"]["print_type"], PENDING_PRINT_METADATA["image"],None,PENDING_PRINT_METADATA["duration"],PENDING_PRINT_METADATA["title"],PRINTER_STATE["print"]["job_id"])

        PENDING_PRINT_METADATA["ams_mapping"] = []
        PENDING_PRINT_METADATA["filamentChanges"] = []
        PENDING_PRINT_METADATA["complete"] = False
        PENDING_PRINT_METADATA["print_id"] = print_id

        for id, filament in PENDING_PRINT_METADATA["filaments"].items():
          insert_filament_usage(print_id, filament["type"], filament["color"], filament["used_g"], id)

        #TODO 
    
      # When stage changed to "change filament" and PENDING_PRINT_METADATA is set
      if (PENDING_PRINT_METADATA and 
          (
            ("stg_cur" in PRINTER_STATE["print"] and (int(PRINTER_STATE["print"]["stg_cur"]) == 4) and      # change filament stage (beginning of print)
              ( 
                "stg_cur" not in PRINTER_STATE_LAST["print"] or                                           # last stage not known
                (
                  PRINTER_STATE_LAST["print"]["stg_cur"] != PRINTER_STATE["print"]["stg_cur"]             # stage has changed and last state was 255 (retract to ams)
                  and "ams" in PRINTER_STATE_LAST["print"] and int(PRINTER_STATE_LAST["print"]["ams"]["tray_tar"]) == 255
                )
                or "ams" not in PRINTER_STATE_LAST["print"]                                               # ams not set in last state
              )
            )
            or                                                                                            # filament changes during printing are in mc_print_sub_stage
            (
              "mc_print_sub_stage" in PRINTER_STATE_LAST["print"] and int(PRINTER_STATE_LAST["print"]["mc_print_sub_stage"]) == 4  # last state was change filament
              and int(PRINTER_STATE["print"]["mc_print_sub_stage"]) == 2                                                           # current state 
            )
            or (
              "ams" in PRINTER_STATE["print"] and int(PRINTER_STATE["print"]["ams"]["tray_tar"]) == 254
            )
            or 
            (
              int(PRINTER_STATE["print"]["stg_cur"]) == 24 and int(PRINTER_STATE_LAST["print"]["stg_cur"]) == 13
            )

          )
      ):
        if "ams" in PRINTER_STATE["print"] and map_filament(int(PRINTER_STATE["print"]["ams"]["tray_tar"])):
            PENDING_PRINT_METADATA["complete"] = True
          

    if PENDING_PRINT_METADATA and PENDING_PRINT_METADATA["complete"]:
      spendFilaments(PENDING_PRINT_METADATA)

      PENDING_PRINT_METADATA = {}

def insert_manual_print(local_path, custom_datetime):
    """
    Traite un fichier .3mf local uploadé manuellement, extrait les métadonnées
    et insère un print et ses filaments dans la base.

    Args:
        local_path (str): Chemin local vers le fichier .3mf
        custom_datetime (datetime.datetime): Date de l'impression définie manuellement

    Returns:
        dict: Résultat avec print_id ou message d'erreur
    """
    try:
        metadata = getMetaDataFrom3mf(f"local:{local_path}", "manual_task")

        if not metadata:
            return {"error": "Échec de l'extraction des métadonnées."}

        name = metadata.get("file", "print")
        if metadata.get("title"):
            name = metadata["title"]
        if metadata.get("plateID") != '1':
            name += f" - {metadata['plateID']}"

        print_id = insert_print(
            name,
            "manual",
            metadata.get("image"),
            custom_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            float(metadata.get("duration", 0)),
            0
        )

        metadata["print_id"] = print_id

        for extruder_id, filament in metadata.get("filaments", {}).items():
            insert_filament_usage(
                print_id,
                filament["type"],
                filament["color"],
                float(filament["used_g"]),
                extruder_id
            )

        return {"success": True, "print_id": print_id}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


def publish(client, msg):
  PRINTER_ID=get_app_setting("PRINTER_ID","")
  result = client.publish(f"device/{PRINTER_ID}/request", json.dumps(msg))
  status = result[0]
  if status == 0:
    logger.info(f"Sent {msg} to topic device/{PRINTER_ID}/request")
    return True

  logger.info(f"Failed to send message to topic device/{PRINTER_ID}/request")
  return False
  
def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convertit une couleur hexadécimale en RGB, ignore l'alpha si présent."""
    hex_color = hex_color.lstrip('#')[:6]
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Convertit RGB (0-255) en CIELAB."""
    # Convert to [0, 1]
    r /= 255
    g /= 255
    b /= 255

    # sRGB gamma correction
    def gamma_correct(c):
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

    r = gamma_correct(r)
    g = gamma_correct(g)
    b = gamma_correct(b)

    # RGB to XYZ
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    # Normalize for D65
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    # XYZ to Lab
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
    """Calcule la distance DeltaE (CIELAB) entre deux couleurs hexadécimales."""
    rgb1 = hex_to_rgb(hex1)
    rgb2 = hex_to_rgb(hex2)
    lab1 = rgb_to_lab(*rgb1)
    lab2 = rgb_to_lab(*rgb2)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))
    
_DATA_DIR = Path("data")
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_MILESTONES_JSON = _DATA_DIR / "milestones_state.json"

# Verrous
_STORE_LOCK = Lock()

def _read_store() -> Dict[str, Any]:
    with _STORE_LOCK:
        if not _MILESTONES_JSON.exists():
            return {}
        try:
            return json.loads(_MILESTONES_JSON.read_text(encoding="utf-8"))
        except Exception:
            # JSON corrompu ? On repart proprement.
            return {}

def _write_store(store: Dict[str, Any]) -> None:
    with _STORE_LOCK:
        tmp = _MILESTONES_JSON.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(_MILESTONES_JSON)  # rename atomique

def _get_job_state_from_store(job_id: str) -> Dict[str, Any]:
    store = _read_store()
    return store.get(job_id) or {}

def _set_job_state_in_store(job_id: str, state: Dict[str, Any]) -> None:
    store = _read_store()
    store[job_id] = state
    _write_store(store)

def _delete_job_state_from_store(job_id: str) -> None:
    store = _read_store()
    if job_id in store:
        del store[job_id]
        _write_store(store)

_MILESTONE_STATE: Dict[str, dict] = {}

def _state(job_id: str) -> dict:
    st = _MILESTONE_STATE.get(job_id)
    if st is None:
        # Fabrique par défaut
        st = {
            "last": -1.0,
            "m50": False, "m99": False, "m100": False,
            "l2": False, "l3": False,
            "last_layer": None,
            "hydrated": False,
            "_print_file": None,
        }
        # Hydrate à partir du JSON si présent
        persisted = _get_job_state_from_store(job_id)
        if persisted:
            st.update(persisted)
            st["hydrated"] = True
        _MILESTONE_STATE[job_id] = st
    return st

def _persist(job_id: str) -> None:
    st = _MILESTONE_STATE.get(job_id)
    if st is None: 
        return
    # On n’écrit pas des clés purement volatiles si besoin, mais ici on garde tout (ça simplifie la reprise)
    _set_job_state_in_store(job_id, st)

def _mark_skip_below_current(job_id: str, st: dict, progress: float|None, layer: int|None) -> None:
    """
    À la première observation d’un job en cours (après reboot ou (re)connexion),
    on marque comme déjà passés TOUS les milestones STRICTEMENT inférieurs à l’état courant.
    => ainsi on ne déclenche rien 'en dessous' de l’état actuel.
    """
    if st.get("_skip_below_done"):
        return

    # Pourcentages
    if isinstance(progress, (int, float)):
        if progress >= 50 and not st["m50"]:
            st["m50"] = True
        if progress >= 99 and not st["m99"]:
            st["m99"] = True
        # m100 ne doit être marqué qu’à 100 réel.
        # On met aussi à jour le 'last' pour ne pas re-déclencher en arrière.
        st["last"] = max(st.get("last", -1.0), float(progress))

    # Couches (fin couche 1 => layer>=2 ; fin couche 2 => layer>=3)
    if isinstance(layer, int):
        if layer >= 2 and not st["l2"]:
            st["l2"] = True
        if layer >= 3 and not st["l3"]:
            st["l3"] = True
        st["last_layer"] = layer

    st["_skip_below_done"] = True  # ne le fait qu’une fois
    _persist(job_id)
    
def _maybe_reset_state_for_new_print(job_id: str, st: dict, fields: dict):
    prog = fields.get("progress")
    status = (fields.get("status") or "").upper()
    print_file = fields.get("print_file")

    def _do_reset():
        st.clear()
        st.update({
            "last": -1.0,
            "m50": False, "m99": False, "m100": False,
            "l2": False, "l3": False,
            "last_layer": None,
            "hydrated": False,
            "_print_file": None,
            "_skip_below_done": False,
        })
        _persist(job_id)

    # Régression forte → nouveau run
    try:
        if isinstance(prog, (int, float)) and isinstance(st.get("last"), (int, float)):
            if st["last"] >= 5.0 and float(prog) <= 1.0:
                _do_reset()
                return
    except Exception:
        pass

    # PREPARE/RUNNING à faible % alors qu’on avait avancé
    try:
        if status in {"PREPARE", "RUNNING"} and isinstance(prog, (int, float)) and prog <= 1.0 and st.get("last", -1.0) > 5.0:
            _do_reset()
            return
    except Exception:
        pass

    # Changement de fichier (si dispo)
    if print_file and st.get("_print_file") and st["_print_file"] != print_file:
        _do_reset()
        return

    if print_file:
        st["_print_file"] = print_file
        _persist(job_id)

def safe_update_status(data):
    # ---------- Utils ----------
    def _to_int(val):
        try:
            return int(val)
        except (TypeError, ValueError):
            try:
                return int(float(val))
            except (TypeError, ValueError):
                return None

    def _split_low_high(val):
        """Décode un 32 bits: low word = courant, high word = cible."""
        v = _to_int(val)
        if v is None:
            return None, None
        current = v & 0xFFFF
        target = (v >> 16) & 0xFFFF
        return current, target

    # ---------- Mapping buse ↔ côté physique (H2) ----------
    LEFT_NOZZLE_ID  = 1  # gauche
    RIGHT_NOZZLE_ID = 0  # droite

    # ---------- Champs de base ----------
    fields = {
        "status": data.get("gcode_state"),
        "progress": data.get("mc_percent"),
        "print_file": data.get("gcode_file"),
        "print_name": data.get("subtask_name"),
        "print_layer": data.get("layer_num"),
        "total_layers": data.get("total_layer_num"),
        "remaining_time": data.get("mc_remaining_time"),
        "tray_now": data.get("ams", {}).get("tray_now"),
    }

    # ---------- BED TEMP (nouveaux firmwares: device.bed.info.temp 32 bits) ----------
    bed_temp_raw = data.get("device", {}).get("bed", {}).get("info", {}).get("temp")
    if bed_temp_raw is not None:
        bed_cur, bed_tgt = _split_low_high(bed_temp_raw)
        if bed_cur is not None:
            fields["bed_temp"] = bed_cur
        if bed_tgt is not None:
            fields["target_bed_temp"] = bed_tgt
    else:
        bt = _to_int(data.get("bed_temper"))
        btt = _to_int(data.get("bed_target_temper"))
        if bt is not None:
            fields["bed_temp"] = bt
        if btt is not None:
            fields["target_bed_temp"] = btt

    # ---------- CHAMBER TEMP (expose la target si présente) ----------
    chamber_temp_raw = data.get("device", {}).get("ctc", {}).get("info", {}).get("temp")
    if chamber_temp_raw is not None:
        c_cur, c_tgt = _split_low_high(chamber_temp_raw)
        if c_cur is not None:
            fields["chamber_temp"] = c_cur
        if c_tgt not in (None, 0):
            fields["target_chamber_temp"] = c_tgt
    else:
        ct = _to_int(data.get("chamber_temper"))
        if ct is not None:
            fields["chamber_temp"] = ct
        ctt = _to_int(data.get("chamber_target_temper"))
        if ctt not in (None, 0):
            fields["target_chamber_temp"] = ctt

    # ---------- EXTRUDER / NOZZLES ----------
    # state: bits 0..3 = nb d’extrudeurs ; bits 4..7 = index actif
    extruder_state = (
        data.get("device", {}).get("extruder", {}).get("state")
        if isinstance(data.get("device", {}).get("extruder", {}), dict)
        else None
    )
    if extruder_state is None:
        extruder_state = data.get("extruder", {}).get("state")

    active_nozzle_index = ((extruder_state >> 4) & 0xF) if extruder_state is not None else 0
    fields["active_nozzle"] = active_nozzle_index

    # Nouveaux firmwares (dual) : liste d’objets {id, temp} sous device.extruder.info
    dev_extr_info = data.get("device", {}).get("extruder", {}).get("info")
    parsed_nozzles = False

    if isinstance(dev_extr_info, list) and dev_extr_info:
        by_id = {}
        for entry in dev_extr_info:
            if isinstance(entry, dict):
                nid = _to_int(entry.get("id"))
                if nid is not None:
                    cur, tgt = _split_low_high(entry.get("temp"))
                    by_id[nid] = (cur, tgt)

        left_cur,  left_tgt  = by_id.get(LEFT_NOZZLE_ID,  (None, None))
        right_cur, right_tgt = by_id.get(RIGHT_NOZZLE_ID, (None, None))

        if left_cur is not None:
            fields["nozzle_left_temp"] = left_cur
        if left_tgt is not None:
            fields["target_nozzle_left_temp"] = left_tgt
        if right_cur is not None:
            fields["nozzle_right_temp"] = right_cur
        if right_tgt is not None:
            fields["target_nozzle_right_temp"] = right_tgt

        # Compat: nozzle_temp/target_nozzle_temp = buse active (fallback: gauche puis droite)
        active_cur = active_tgt = None
        if active_nozzle_index == LEFT_NOZZLE_ID and left_cur is not None:
            active_cur, active_tgt = left_cur, left_tgt
        elif active_nozzle_index == RIGHT_NOZZLE_ID and right_cur is not None:
            active_cur, active_tgt = right_cur, right_tgt
        elif left_cur is not None:
            active_cur, active_tgt = left_cur, left_tgt
        elif right_cur is not None:
            active_cur, active_tgt = right_cur, right_tgt

        if active_cur is not None:
            fields["nozzle_temp"] = active_cur
        if active_tgt is not None:
            fields["target_nozzle_temp"] = active_tgt

        parsed_nozzles = True

    # Cas mono-buse (nouveaux firmwares top-level)
    if not parsed_nozzles:
        top_extr_info = data.get("extruder", {}).get("info", {})
        if isinstance(top_extr_info, dict) and "temp" in top_extr_info:
            cur, tgt = _split_low_high(top_extr_info.get("temp"))
            if cur is not None:
                fields["nozzle_temp"] = cur
            if tgt is not None:
                fields["target_nozzle_temp"] = tgt
            parsed_nozzles = True

    # Fallback très rétro: anciens champs plats
    if not parsed_nozzles:
        nt = _to_int(data.get("nozzle_temper"))
        ntt = _to_int(data.get("nozzle_target_temper"))
        if nt is not None:
            fields["nozzle_temp"] = nt
        if ntt is not None:
            fields["target_nozzle_temp"] = ntt

    # ---------- AMS / TRAY MAPPING ----------
    try:
        tray_now = int(data.get("ams", {}).get("tray_now"))
    except (TypeError, ValueError):
        tray_now = None

    ams_list = data.get("ams", {}).get("ams", [])
    fields["ams"] = {}
    for ams in ams_list:
        ams_id = int(ams.get("id"))
        dry_time = int(ams.get("dry_time", 0))
        fields["ams"][ams_id] = {"dry_time": dry_time}
    fields["tray_local_id"] = None
    fields["tray_ams_id"] = None

    # Map AMS -> extrudeur (aligne avec LEFT/RIGHT_* ci-dessus)
    ams_extruder_map = {0: RIGHT_NOZZLE_ID, 1: LEFT_NOZZLE_ID}

    if tray_now is not None and isinstance(ams_list, list) and tray_now != 255:
        candidate_trays = []
        for ams in ams_list:
            try:
                ams_id = int(ams.get("id"))
            except (TypeError, ValueError):
                continue
            for tray in ams.get("tray", []):
                try:
                    tray_id = int(tray.get("id"))
                except (TypeError, ValueError):
                    continue
                if tray_id == tray_now:
                    candidate_trays.append((ams_id, tray_id))

        if len(ams_list) == 1 and candidate_trays:
            fields["tray_ams_id"], fields["tray_local_id"] = candidate_trays[0]
        elif len(ams_list) > 1:
            for ams_id, tray_id in candidate_trays:
                if ams_extruder_map.get(ams_id) == active_nozzle_index:
                    fields["tray_ams_id"] = ams_id
                    fields["tray_local_id"] = tray_id
                    break
        if fields["tray_ams_id"] is None and candidate_trays:
            fields["tray_ams_id"], fields["tray_local_id"] = candidate_trays[0]

    elif tray_now is not None and isinstance(ams_list, list) and tray_now == 255:
        # Bobine externe
        fields["tray_ams_id"] = 255
        fields["tray_local_id"] = 0

    # ---------- Temps restant / ETA ----------
    remaining = fields.get("remaining_time")
    if isinstance(remaining, (int, float)):
        if remaining > 0:
            estimated_end = datetime.now() + timedelta(minutes=remaining)
            fields["estimated_end"] = estimated_end.strftime("%H:%M")
        hours = int(remaining // 60)
        minutes = int(remaining % 60)
        fields["remaining_time_str"] = f"{hours}h {minutes:02d}min" if hours > 0 else f"{minutes}min"
    ####----snapshot-----
    try:
        job_id = data.get("job_id")
        if not job_id:
            # Certaines trames n’ont pas de job_id : on ignore silencieusement
            return
        job_id = str(job_id)

        prog_raw = fields.get("progress")
        layer_raw = fields.get("print_layer")

        # Parsing robustes
        try:
            prog = float(prog_raw) if prog_raw is not None else None
        except Exception:
            prog = None

        layer = None
        if layer_raw is not None:
            with suppress(Exception):
                layer = int(layer_raw)

        st = _state(job_id)
        # Nouveau run ?
        _maybe_reset_state_for_new_print(job_id, st, fields)

        # À la première observation d’un job en cours (ex: reboot), on fige l'état bas
        # => ne jamais déclencher en dessous de l'état courant
        _maybe_reset_state_for_new_print(job_id, st, fields)

        # Seulement si l’on accroche un job déjà entamé (reboot/reconnect),
        # on marque les milestones <= état courant comme "déjà passés".
        is_mid_print_attach = (
            (st.get("last", -1.0) <= 0) and (
                (isinstance(prog, (int, float)) and prog >= 1.0) or
                (isinstance(layer, int) and layer >= 2)
            )
        )
        if is_mid_print_attach and not st.get("_skip_below_done"):
            _mark_skip_below_current(job_id, st, prog, layer)
        # 1) Milestones % : 50 / 99 / 100
        if prog is not None and 0.0 <= prog <= 100.0:
            # Calcul des paliers à tirer (en évitant ceux déjà marqués true)
            to_fire = []
            prev = st.get("last", -1.0)
            # On déclenche lors d’un passage 'vers le haut'
            def _cross(prev_v, cur_v, th):
                return prev_v < th <= cur_v

            if not st["m50"] and _cross(prev, prog, 50.0):
                to_fire.append(50)
            if not st["m99"] and _cross(prev, prog, 99.0):
                to_fire.append(99)
            if not st["m100"] and _cross(prev, prog, 100.0):
                to_fire.append(100)

            for pct in to_fire:
                fire_and_forget(
                    snapshot_milestone, job_id, pct,
                    basename=f"Impression-{pct}",
                    name=f"snapshot-{job_id}-{pct}"
                )
                logger.debug(f"[milestones] FIRE% {pct} for job={job_id} (prog={prog}, prev={prev})")
                if pct == 50:  st["m50"]  = True
                if pct == 99:  st["m99"]  = True
                if pct == 100: st["m100"] = True

            # Mémorise dernier %
            st["last"] = max(prev, prog)
            _persist(job_id)

        # 2) Milestones fin couche 1 & 2 :
        #    - fin couche 1  => quand layer atteint pour la 1ère fois 2
        #    - fin couche 2  => quand layer atteint pour la 1ère fois 3
        if layer is not None:
            prev_layer = st.get("last_layer") or 0

            if not st["l2"] and prev_layer < 2 <= layer:
                fire_and_forget(
                    snapshot_milestone, job_id, 0,
                    basename="Impression-couche-1",
                    name=f"snapshot-{job_id}-L1"
                )
                logger.debug(f"[milestones] FIRE layer1 (layer={layer}) for job={job_id}")
                st["l2"] = True

            if not st["l3"] and prev_layer < 3 <= layer:
                fire_and_forget(
                    snapshot_milestone, job_id, 0,
                    basename="Impression-couche-2",
                    name=f"snapshot-{job_id}-L2"
                )
                logger.debug(f"[milestones] FIRE layer2 (layer={layer}) for job={job_id}")
                st["l3"] = True

            st["last_layer"] = max(prev_layer, layer)
            _persist(job_id)

    except Exception:
        logger.debug("Milestones snapshot: erreur non bloquante", exc_info=True)
    # ---------- Détection fin/échec (antirebond) ----------
    job_id = data.get("job_id")
    status = (fields.get("status") or "").upper()
    if job_id and status in {"FINISH", "FAILED"}:
        now = time.time()
        if job_id not in PROCESSED_JOBS:
            if job_id not in PENDING_JOBS:
                PENDING_JOBS[job_id] = (status, now)
            else:
                prev_status, first_seen = PENDING_JOBS[job_id]
                if prev_status == status:
                    if now - first_seen >= 10:
                        final_status = "SUCCESS" if status == "FINISH" else "FAILED"
                        if status == 'FAILED':
                            pct=int(fields.get("progress"))
                            snapshot_milestone(job_id, pct, basename=f"Impression-{pct}-echec")
                        update_print_field_with_job_id(job_id, "status", final_status)
                        PROCESSED_JOBS.add(job_id)
                        PENDING_JOBS.pop(job_id, None)
                        with suppress(Exception):
                            _MILESTONE_STATE.pop(str(job_id), None)
                            _delete_job_state_from_store(str(job_id))
                else:
                    PENDING_JOBS[job_id] = (status, now)

    # ---------- Publication ----------
    update_status({k: v for k, v in fields.items() if v is not None})

# Inspired by https://github.com/Donkie/Spoolman/issues/217#issuecomment-2303022970
def on_message(client, userdata, msg):
  global LAST_AMS_CONFIG, PRINTER_STATE, PRINTER_STATE_LAST, PENDING_PRINT_METADATA, PRINTER_MODEL
  
  try:
    topic = msg.topic
    data = json.loads(msg.payload.decode())
    
    if "print" in data:
      append_to_rotating_file("/home/app/logs/mqtt.log", msg.payload.decode())
      if "report" in topic:
        try:
            safe_update_status(data["print"])
        except Exception as e:
            traceback.print_exc()
      if AUTO_SPEND:
          update_dict(PRINTER_STATE, data)
          if ("command" in data["print"] and data["print"]["command"] == "project_file" and "url" in data["print"]) or (( "print_type" in PRINTER_STATE["print"] and PRINTER_STATE["print"]["print_type"] == "local" and "print" in PRINTER_STATE_LAST)):
          # Lance processMessage en thread si pas déjà en cours
            if PROCESSMSG_LOCK.acquire(blocking=False):
                fire_and_forget(processMessage, data, name="processMessage", release_lock=True)
            else:
                logger.debug("[async] processMessage déjà en cours — skip")
          PRINTER_STATE_LAST = copy.deepcopy(PRINTER_STATE)
      
    # Save external spool tray data
    if "print" in data and "vt_tray" in data["print"]:
      LAST_AMS_CONFIG["vt_tray"] = data["print"]["vt_tray"]

    # Save ams spool data
    if "print" in data and "ams" in data["print"] and "ams" in data["print"]["ams"]:
      LAST_AMS_CONFIG["ams"] = data["print"]["ams"]["ams"]
      for ams in data["print"]["ams"]["ams"]:
        #logger.info(f"AMS [{num2letter(ams['id'])}] (hum: {ams['humidity_raw']}, temp: {ams['temp']}ºC)")
        spools = fetch_spools()
        for tray in ams["tray"]:
          if "tray_sub_brands" in tray:
            #logger.info(f"    - [{num2letter(ams['id'])}{tray['id']}] {tray['tray_sub_brands']} {tray['tray_color']} ({str(tray['remain']).zfill(3)}%) [[{tray['tray_uuid']}]] [[{tray['tray_info_idx']}]]")

            foundspool = None
            foundMode =''
            tray_uuid = tray["tray_uuid"]
            tray_info_idx = tray["tray_info_idx"]
            tray_color = tray["tray_color"]
            tag='n/a'
            filament_id='n/a'
            mapped_spool_id = get_tray_spool_map(tray_uuid, tray_info_idx, tray_color)

            if mapped_spool_id:
                spool_match = next((s for s in spools if s["id"] == mapped_spool_id), None)
                if spool_match:
                    foundMode='Manual'
                    foundspool = spool_match
                else:
                    delete_tray_spool_map_by_id(mapped_spool_id)
            if not foundspool:
                for spool in spools:
                    if not spool.get("extra", {}).get("tag") and not spool.get("filament", {}).get("extra",{}).get("filament_id"):
                        continue
                    if spool.get("extra", {}).get("tag"):
                        tag = spool["extra"]["tag"]
                    if spool.get("filament", {}).get("extra",{}).get("filament_id"):
                        filament_id = spool["filament"]["extra"]["filament_id"]
                    if tag != tray["tray_uuid"] and filament_id != tray["tray_info_idx"]:
                        continue
                    if tray_uuid == tag:
                        #logger.info('Found spool with tag')
                        foundspool= spool
                        foundMode='Tag'
                        break
                    else:
                        if spool.get("filament", {}).get("extra",{}).get("filament_id"):
                            color_dist = color_distance(spool["filament"]["color_hex"],tray['tray_color'])
                            spool['color_dist']=color_dist
                            #logger.info(filament_id + ' ' +spool["filament"]["color_hex"] + ' : ' + str(color_dist)) 
                            if foundspool == None:
                                if color_dist<50:
                                    foundMode='Profile'
                                    foundspool= spool
                            else:
                                if color_dist<foundspool['color_dist']:
                                    foundMode='Profile'
                                    foundspool= spool
            if foundspool == None:
              logger.info("      - Not found. Update spool tag or filament_id and color!")
              clearActiveTray(ams['id'], tray["id"])
            else:
                setActiveTray(foundspool['id'], ams['id'], tray["id"],foundMode)
          else:
              clearActiveTray(ams['id'], tray["id"])
  except Exception as e:
    traceback.print_exc()

def on_connect(client, userdata, flags, rc):
  PRINTER_ID=get_app_setting("PRINTER_ID","")
  global MQTT_CLIENT_CONNECTED
  MQTT_CLIENT_CONNECTED = True
  logger.info("Connected with result code " + str(rc))
  client.subscribe(f"device/{PRINTER_ID}/report")
  publish(client, GET_VERSION)
  publish(client, PUSH_ALL)

def on_disconnect(client, userdata, rc):
  global MQTT_CLIENT_CONNECTED
  MQTT_CLIENT_CONNECTED = False
  logger.info("Disconnected with result code " + str(rc))
  
MQTT_LOCK = Lock()

def async_subscribe():
    global MQTT_CLIENT
    global MQTT_CLIENT_CONNECTED

    MQTT_CLIENT_CONNECTED = False

    while True:
        with MQTT_LOCK:
            while not MQTT_CLIENT_CONNECTED:
                try:
                    # 🔁 Fermer l'ancien client s'il existe
                    if MQTT_CLIENT is not None:
                        try:
                            MQTT_CLIENT.loop_stop()
                            MQTT_CLIENT.disconnect()
                        except:
                            pass

                    # 🔁 Récupération dynamique des paramètres
                    printer_code = get_app_setting("PRINTER_ACCESS_CODE", default='')
                    printer_ip = get_app_setting("PRINTER_IP", default='')

                    MQTT_CLIENT = mqtt.Client()
                    MQTT_CLIENT.username_pw_set("bblp", printer_code)

                    ssl_ctx = ssl.create_default_context()
                    ssl_ctx.check_hostname = False
                    ssl_ctx.verify_mode = ssl.CERT_NONE

                    MQTT_CLIENT.tls_set_context(ssl_ctx)
                    MQTT_CLIENT.tls_insecure_set(True)

                    MQTT_CLIENT.on_connect = on_connect
                    MQTT_CLIENT.on_disconnect = on_disconnect
                    MQTT_CLIENT.on_message = on_message

                    logger.info("🔄 Trying to connect ...")
                    MQTT_CLIENT.connect(printer_ip, 8883, MQTT_KEEPALIVE)
                    MQTT_CLIENT.loop_start()
                    logger.info("Connected ...")

                except Exception as e:
                    logger.info(f"⚠️ connection failed: {e}, new try in 15 seconds...")

                time.sleep(15)

        time.sleep(15)

def init_mqtt():
  # Start the asynchronous processing in a separate thread
  thread = Thread(target=async_subscribe)
  thread.start()

def getLastAMSConfig():
  global LAST_AMS_CONFIG
  return LAST_AMS_CONFIG


def getMqttClient():
  global MQTT_CLIENT
  return MQTT_CLIENT

def isMqttClientConnected():
  global MQTT_CLIENT_CONNECTED
  return MQTT_CLIENT_CONNECTED