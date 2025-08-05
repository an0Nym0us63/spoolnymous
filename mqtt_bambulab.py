import json
import ssl
import traceback
from threading import Thread

from datetime import datetime, timedelta
import paho.mqtt.client as mqtt

from config import PRINTER_ID, PRINTER_CODE, PRINTER_IP, AUTO_SPEND, EXTERNAL_SPOOL_AMS_ID, EXTERNAL_SPOOL_ID
from messages import GET_VERSION, PUSH_ALL
from spoolman_service import spendFilaments, setActiveTray, fetchSpools
from tools_3mf import getMetaDataFrom3mf
import time
import copy
import math
from collections.abc import Mapping
from logger import append_to_rotating_file
from print_history import  insert_print, insert_filament_usage, update_filament_spool,update_print_status_with_job_id

from globals import PRINTER_STATUS, PRINTER_STATUS_LOCK, PROCESSED_JOBS

MQTT_CLIENT = {}  # Global variable storing MQTT Client
MQTT_CLIENT_CONNECTED = False
MQTT_KEEPALIVE = 60
LAST_AMS_CONFIG = {}  # Global variable storing last AMS configuration

PRINTER_STATE = {}
PRINTER_STATE_LAST = {}

PENDING_PRINT_METADATA = {}

def update_status(new_data):
    with PRINTER_STATUS_LOCK:
        PRINTER_STATUS.update(new_data)
        
def getPrinterModel():
    global PRINTER_ID
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
    print(f'Filamentchange {len(PENDING_PRINT_METADATA["filamentChanges"])}: Tray {tray_tar}')

    # Anzahl der erkannten Wechsel
    change_count = len(PENDING_PRINT_METADATA["filamentChanges"]) - 1  # -1, weil der erste Eintrag kein Wechsel ist

    # Slot in der Wechselreihenfolge bestimmen
    for tray, usage_count in PENDING_PRINT_METADATA["filamentOrder"].items():
        if usage_count == change_count:
            PENDING_PRINT_METADATA["ams_mapping"].append(tray_tar)
            print(f"✅ Tray {tray_tar} assigned Filament to {tray}")

            for filament, tray in enumerate(PENDING_PRINT_METADATA["ams_mapping"]):
              print(f"  Filament {filament} → Tray {tray}")


    # Falls alle Slots zugeordnet sind, Ausgabe der Zuordnung
    if len(PENDING_PRINT_METADATA["ams_mapping"]) == len(PENDING_PRINT_METADATA["filamentOrder"]):
        print("\n✅ All trays assigned:")
        for filament, tray in enumerate(PENDING_PRINT_METADATA["ams_mapping"]):
            print(f"  Filament {tray} → Tray {tray}")

        return True
  
  return False
  
def processMessage(data):
  global LAST_AMS_CONFIG, PRINTER_STATE, PRINTER_STATE_LAST, PENDING_PRINT_METADATA
    
   # Prepare AMS spending estimation
  if "print" in data:
    update_dict(PRINTER_STATE, data)
    #print(str(data))
    if "command" in data["print"] and data["print"]["command"] == "project_file" and "url" in data["print"]:
      PENDING_PRINT_METADATA = getMetaDataFrom3mf(data["print"]["url"],data["print"]["subtask_name"])
      name=PRINTER_STATE["print"]["subtask_name"]
      if PENDING_PRINT_METADATA["title"] != '':
        name = PENDING_PRINT_METADATA["title"]
      if (PENDING_PRINT_METADATA["plateID"] != '1'):
        name += ' - ' +PENDING_PRINT_METADATA["plateID"]
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
  
    PRINTER_STATE_LAST = copy.deepcopy(PRINTER_STATE)

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
  result = client.publish(f"device/{PRINTER_ID}/request", json.dumps(msg))
  status = result[0]
  if status == 0:
    print(f"Sent {msg} to topic device/{PRINTER_ID}/request")
    return True

  print(f"Failed to send message to topic device/{PRINTER_ID}/request")
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

def safe_update_status(data):
    fields = {
        "status": data.get("gcode_state"),
        "progress": data.get("mc_percent"),
        "bed_temp": data.get("bed_temper"),
        "nozzle_temp": data.get("nozzle_temper"),
        "print_file": data.get("gcode_file"),
        "print_name": data.get("subtask_name"),
        "print_layer": data.get("layer_num"),
        "total_layers": data.get("total_layer_num"),
        "remaining_time": data.get("mc_remaining_time"),
        "chamber_temp": data.get("chamber_temper"),
        "tray_now": data.get("ams",{}).get("tray_now")
    }
    tray_now_raw = data.get("ams", {}).get("tray_now")
    try:
        tray_now = int(tray_now_raw)
    except (TypeError, ValueError):
        tray_now = None
    
    extruder_state = data.get("device", {}).get("extruder", {}).get("state")
    active_nozzle_index = (extruder_state >> 4) & 0xF if extruder_state is not None else 0
    
    ams_list = data.get("ams", {}).get("ams", [])
    fields["tray_local_id"] = None
    fields["tray_ams_id"] = None
    
    # Mapping logique AMS ↔ extrudeur (à adapter si nécessaire)
    ams_extruder_map = {
        0: 0,  # AMS 0 → extrudeur gauche
        1: 1   # AMS 1 → extrudeur droit
    }
    
    if tray_now is not None and isinstance(ams_list, list):
        candidate_trays = []
    
        for ams in ams_list:
            try:
                ams_id = int(ams.get("id"))
            except (TypeError, ValueError):
                continue
            trays = ams.get("tray", [])
            for tray in trays:
                try:
                    tray_id = int(tray.get("id"))
                except (TypeError, ValueError):
                    continue
                if tray_id == tray_now:
                    candidate_trays.append((ams_id, tray_id))
    
        # Si un seul AMS → pas de conflit
        if len(ams_list) == 1 and candidate_trays:
            fields["tray_ams_id"], fields["tray_local_id"] = candidate_trays[0]
    
        # Si plusieurs AMS → chercher celui correspondant à l'extrudeur actif
        elif len(ams_list) > 1:
            for ams_id, tray_id in candidate_trays:
                if ams_extruder_map.get(ams_id) == active_nozzle_index:
                    fields["tray_ams_id"] = ams_id
                    fields["tray_local_id"] = tray_id
                    break
    
        # Fallback : si rien trouvé, on prend le premier match
        if fields["tray_ams_id"] is None and candidate_trays:
            fields["tray_ams_id"], fields["tray_local_id"] = candidate_trays[0]

    remaining = fields.get("remaining_time")
    if isinstance(remaining, (int, float)):
        if remaining > 0:
            # Heure d'arrivée estimée
            estimated_end = datetime.now() + timedelta(minutes=remaining)
            fields["estimated_end"] = estimated_end.strftime("%H:%M")
    
            # Format heure/minute lisible
            hours = int(remaining // 60)
            minutes = int(remaining % 60)
            if hours > 0:
                fields["remaining_time_str"] = f"{hours}h {minutes:02d}min"
            else:
                fields["remaining_time_str"] = f"{minutes}min"
        else:
            job_id = data.get("job_id")
            status = (fields.get("status") or "").upper()
        
            if job_id and job_id not in PROCESSED_JOBS:
                if status == "FAILED"
                    update_print_status_with_job_id(job_id, "status", "FAILED")
                    PROCESSED_JOBS.add(job_id)
                elif status == "FINISHED":
                    update_print_status_with_job_id(job_id, "status", "SUCCESS")
                    PROCESSED_JOBS.add(job_id)
    update_status({k: v for k, v in fields.items() if v is not None})

# Inspired by https://github.com/Donkie/Spoolman/issues/217#issuecomment-2303022970
def on_message(client, userdata, msg):
  global LAST_AMS_CONFIG, PRINTER_STATE, PRINTER_STATE_LAST, PENDING_PRINT_METADATA, PRINTER_MODEL
  
  try:
    topic = msg.topic
    data = json.loads(msg.payload.decode())
    try:
        if "report" in topic and "print" in data:
            safe_update_status(data["print"])
    except Exception as e:
        traceback.print_exc()
    if "print" in data:
      append_to_rotating_file("/home/app/logs/mqtt.log", msg.payload.decode())

    if AUTO_SPEND:
        processMessage(data)
      
    # Save external spool tray data
    if "print" in data and "vt_tray" in data["print"]:
      LAST_AMS_CONFIG["vt_tray"] = data["print"]["vt_tray"]

    # Save ams spool data
    if "print" in data and "ams" in data["print"] and "ams" in data["print"]["ams"]:
      LAST_AMS_CONFIG["ams"] = data["print"]["ams"]["ams"]
      for ams in data["print"]["ams"]["ams"]:
        #print(f"AMS [{num2letter(ams['id'])}] (hum: {ams['humidity_raw']}, temp: {ams['temp']}ºC)")
        for tray in ams["tray"]:
          if "tray_sub_brands" in tray:
            #print(f"    - [{num2letter(ams['id'])}{tray['id']}] {tray['tray_sub_brands']} {tray['tray_color']} ({str(tray['remain']).zfill(3)}%) [[{tray['tray_uuid']}]] [[{tray['tray_info_idx']}]]")

            foundspool = None
            tray_uuid = "00000000000000000000000000000000"
            tag='n/a'
            filament_id='n/a'
            tray_uuid = tray["tray_uuid"]
            for spool in fetchSpools(True):
              if not spool.get("extra", {}).get("tag") and not spool.get("filament", {}).get("extra",{}).get("filament_id"):
                continue
              if spool.get("extra", {}).get("tag"):
                tag = json.loads(spool["extra"]["tag"])
              if spool.get("filament", {}).get("extra",{}).get("filament_id"):
                filament_id = json.loads(spool["filament"]["extra"]["filament_id"])
              if tag != tray["tray_uuid"] and filament_id != tray["tray_info_idx"]:
                continue
              if tray_uuid == tag:
                #print('Found spool with tag')
                foundspool= spool
                break
              else:
                if spool.get("filament", {}).get("extra",{}).get("filament_id"):
                    color_dist = color_distance(spool["filament"]["color_hex"],tray['tray_color'])
                    spool['color_dist']=color_dist
                    #print(filament_id + ' ' +spool["filament"]["color_hex"] + ' : ' + str(color_dist)) 
                    if foundspool == None:
                        if color_dist<50:
                            foundspool= spool
                    else:
                        if color_dist<foundspool['color_dist']:
                            foundspool= spool

              # TODO: filament remaining - Doesn't work for AMS Lite
              # requests.patch(f"http://{SPOOLMAN_IP}:7912/api/v1/spool/{spool['id']}", json={
              #  "remaining_weight": tray["remain"] / 100 * tray["tray_weight"]
              # })

            if foundspool == None:
              print("      - Not found. Update spool tag or filament_id and color!")
            else:
                #print("Found spool " + str(foundspool))
                setActiveTray(foundspool['id'], foundspool["extra"], ams['id'], tray["id"])
              
  except Exception as e:
    traceback.print_exc()

def on_connect(client, userdata, flags, rc):
  global MQTT_CLIENT_CONNECTED
  MQTT_CLIENT_CONNECTED = True
  print("Connected with result code " + str(rc))
  client.subscribe(f"device/{PRINTER_ID}/report")
  publish(client, GET_VERSION)
  publish(client, PUSH_ALL)

def on_disconnect(client, userdata, rc):
  global MQTT_CLIENT_CONNECTED
  MQTT_CLIENT_CONNECTED = False
  print("Disconnected with result code " + str(rc))
  
def async_subscribe():
  global MQTT_CLIENT
  global MQTT_CLIENT_CONNECTED
  
  MQTT_CLIENT_CONNECTED = False
  MQTT_CLIENT = mqtt.Client()
  MQTT_CLIENT.username_pw_set("bblp", PRINTER_CODE)
  ssl_ctx = ssl.create_default_context()
  ssl_ctx.check_hostname = False
  ssl_ctx.verify_mode = ssl.CERT_NONE
  MQTT_CLIENT.tls_set_context(ssl_ctx)
  MQTT_CLIENT.tls_insecure_set(True)
  MQTT_CLIENT.on_connect = on_connect
  MQTT_CLIENT.on_disconnect = on_disconnect
  MQTT_CLIENT.on_message = on_message
  
  while True:
    while not MQTT_CLIENT_CONNECTED:
      try:
          print("🔄 Trying to connect ...", flush=True)
          MQTT_CLIENT.connect(PRINTER_IP, 8883, MQTT_KEEPALIVE)
          MQTT_CLIENT.loop_start()
          
      except Exception as e:
          print(f"⚠️ connection failed: {e}, new try in 15 seconds...", flush=True)

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