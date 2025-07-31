import json
import traceback
import uuid
import math
import datetime
import os
import re
from collections import defaultdict

import secrets

from flask_login import LoginManager, login_required
from auth import auth_bp, User, get_stored_user

from flask import Flask, request, render_template, redirect, url_for,jsonify,g, make_response,send_from_directory, abort

from config import BASE_URL, AUTO_SPEND, SPOOLMAN_BASE_URL, EXTERNAL_SPOOL_AMS_ID, EXTERNAL_SPOOL_ID, PRINTER_NAME,LOCATION_MAPPING,AMS_ORDER, COST_BY_HOUR
from filament import generate_filament_brand_code, generate_filament_temperatures
from frontend_utils import color_is_dark
from messages import AMS_FILAMENT_SETTING
from mqtt_bambulab import fetchSpools, getLastAMSConfig, publish, getMqttClient, setActiveTray, isMqttClientConnected, init_mqtt, getPrinterModel
from spoolman_client import patchExtraTags, getSpoolById, consumeSpool, archive_spool, reajust_spool
from spoolman_service import augmentTrayDataWithSpoolMan, trayUid, getSettings
from print_history import get_prints_with_filament, update_filament_spool, get_filament_for_slot,get_distinct_values,update_print_filename,get_filament_for_print, delete_print, get_tags_for_print, add_tag_to_print, remove_tag_from_print,update_filament_usage,update_print_history_field,create_print_group,get_print_groups,update_print_group_field,update_group_created_at,get_group_id_of_print,get_statistics,adjustDuration,set_group_primary_print,set_sold_info


COLOR_FAMILIES = {
    # Neutres
    'Black': (0, 0, 0),
    'White': (255, 255, 255),
    'Grey': (160, 160, 160),

    # Rouges et dérivés
    'Red': (220, 20, 60),         # Crimson
    'Dark Red': (139, 0, 0),      # sombre
    'Pink': (255, 182, 193),      # pastel
    'Magenta': (255, 0, 255),     # fuchsia
    'Brown': (150, 75, 0),        # chocolat

    # Jaunes et dérivés
    'Yellow': (255, 220, 0),      # chaud
    'Gold': (212, 175, 55),       # doré
    'Orange': (255, 140, 0),      # foncé

    # Verts
    'Green': (80, 200, 120),      # gazon
    'Dark Green': (0, 100, 0),    # forêt
    'Lime': (191, 255, 0),        # fluo
    'Teal': (0, 128, 128),        # turquoise

    # Bleus et violets
    'Blue': (100, 150, 255),      # clair
    'Navy': (0, 0, 128),          # foncé
    'Cyan': (0, 255, 255),        # turquoise clair
    'Lavender': (230, 230, 250),  # violet pastel
    'Purple': (160, 32, 240),
    'Dark Purple': (90, 60, 120), # violet foncé
}

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')[:6]
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    r /= 255
    g /= 255
    b /= 255

    def gamma_correct(c):
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

    r = gamma_correct(r)
    g = gamma_correct(g)
    b = gamma_correct(b)

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

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
    rgb1 = hex_to_rgb(hex1)
    rgb2 = hex_to_rgb(hex2)
    lab1 = rgb_to_lab(*rgb1)
    lab2 = rgb_to_lab(*rgb2)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


def two_closest_families(hex_color: str, threshold: float = 60.0) -> list[str]:
    """
    Retourne la famille la plus proche et la deuxième si sa distance est < threshold.
    """
    distances = {
        family: color_distance(hex_color, '#{:02X}{:02X}{:02X}'.format(*rgb))
        for family, rgb in COLOR_FAMILIES.items()
    }
    sorted_families = sorted(distances.items(), key=lambda x: x[1])

    result = [sorted_families[0][0]]  # toujours la plus proche
    if sorted_families[1][1] <= threshold:
        result.append(sorted_families[1][0])
    return result

def compute_pagination_pages(page, total_pages, window=2, max_buttons=5):
    pages = []
    if total_pages <= max_buttons:
        pages = list(range(1, total_pages + 1))
    else:
        pages = [1]

        start = max(2, page - window)
        end = min(total_pages - 1, page + window)

        if start > 2:
            pages.append('…')

        pages.extend(range(start, end + 1))

        if end < total_pages - 1:
            pages.append('…')

        pages.append(total_pages)

    return pages

def redirect_back(focus_id=None):
    args = request.form.to_dict(flat=True)
    page = args.get("page") or request.args.get("page") or 1
    kwargs = {"page": page}
    if focus_id:
        kwargs["focus_id"] = focus_id
    return redirect(url_for("print_history", **kwargs))

init_mqtt()

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
login_manager = LoginManager()
login_manager.login_view = 'auth.login'  # redirige vers /login si non connecté
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    data = get_stored_user()
    if data and user_id in data:
        return User(user_id)
    elif user_id == app.config.get("DEFAULT_ADMIN_USERNAME", "admin"):
        return User(user_id)
    return None

@app.before_request
def detect_webview():
    g.is_webview = request.cookies.get('webview') == '1'

@app.before_request
def require_login():
    from flask_login import current_user
    exempt_routes = {'auth.login', 'auth.logout', 'auth.settings', 'auth.autologin_token', 'static'}
    if request.endpoint not in exempt_routes and not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    
@app.template_filter('datetimeformat')
def datetimeformat(value, locale='fr'):
    from datetime import datetime
    dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%d/%m/%Y %H:%M")
    
@app.template_filter('hm_format')
def hm_format(hours: float):
    h = int(hours)
    m = int(round((hours - h) * 60))
    return f"{h}h {m:02d}min"
    
@app.context_processor
def frontend_utilities():
    def url_with_args(**kwargs):
        query = request.args.to_dict(flat=False)
        query.pop('page', None)
        query.update({k: [str(v)] if not isinstance(v, list) else v for k, v in kwargs.items()})
        return url_for(request.endpoint, **query)
    return dict(
        SPOOLMAN_BASE_URL=SPOOLMAN_BASE_URL,
        AUTO_SPEND=AUTO_SPEND,
        color_is_dark=color_is_dark,
        BASE_URL=BASE_URL,
        EXTERNAL_SPOOL_AMS_ID=EXTERNAL_SPOOL_AMS_ID,
        EXTERNAL_SPOOL_ID=EXTERNAL_SPOOL_ID,
        PRINTER_MODEL=getPrinterModel(),
        PRINTER_NAME=PRINTER_NAME,
        url_with_args=url_with_args
    )

@app.route("/issue")
def issue():
  if not isMqttClientConnected():
    return render_template('error.html', exception="MQTT is disconnected. Is the printer online?")
    
  ams_id = request.args.get("ams")
  tray_id = request.args.get("tray")
  if not all([ams_id, tray_id]):
    return render_template('error.html', exception="Missing AMS ID, or Tray ID.")

  fix_ams = None

  spool_list = fetchSpools()
  last_ams_config = getLastAMSConfig()
  if ams_id == EXTERNAL_SPOOL_AMS_ID:
    fix_ams = last_ams_config.get("vt_tray", {})
  else:
    for ams in last_ams_config.get("ams", []):
      if ams["id"] == ams_id:
        fix_ams = ams
        break

  active_spool = None
  for spool in spool_list:
    if spool.get("extra") and spool["extra"].get("active_tray") and spool["extra"]["active_tray"] == json.dumps(trayUid(ams_id, tray_id)):
      active_spool = spool
      break

  #TODO: Determine issue
  #New bambulab spool
  #Tray empty, but spoolman has record
  #Extra tag mismatch?
  #COLor/type mismatch

  return render_template('issue.html', fix_ams=fix_ams, active_spool=active_spool)

@app.route("/fill")
def fill():
  if not isMqttClientConnected():
    return render_template('error.html', exception="MQTT is disconnected. Is the printer online?")
    
  ams_id = request.args.get("ams")
  tray_id = request.args.get("tray")
  if not all([ams_id, tray_id]):
    return render_template('error.html', exception="Missing AMS ID, or Tray ID.")

  spool_id = request.args.get("spool_id")
  if spool_id:
    spool_data = getSpoolById(spool_id)
    setActiveTray(spool_id, spool_data["extra"], ams_id, tray_id)
    setActiveSpool(ams_id, tray_id, spool_data)
    return redirect(url_for('home', success_message=f"Updated Spool ID {spool_id} to AMS {ams_id}, Tray {tray_id}."))
  else:
    page = int(request.args.get("page", 1))
    per_page = 25
    search = request.args.get("search", "").lower()
    sort = request.args.get("sort", "default")

    all_filaments = fetchSpools() or []

    # filtre nom / couleur
    if search:
        search_terms = search.split()

        def matches(f):
            filament = f.get("filament", {})
            vendor = filament.get("vendor", {})
            fields = [
                filament.get("name", "").lower(),
                filament.get("material", "").lower(),
                vendor.get("name", "").lower(),
                f.get("location", "").lower(),
            ]
            # Chaque terme doit être présent dans au moins un des champs
            return all(
                any(term in field for field in fields)
                for term in search_terms
            )

        all_filaments = [f for f in all_filaments if matches(f)]

    def sort_key(f):
        filament = f.get("filament", {})
        vendor = filament.get("vendor", {})
        return (
            f.get("location", "").lower(),
            filament.get("material", "").lower(),
            vendor.get("name", "").lower(),
            filament.get("name", "").lower()
        )
    if sort == "remaining":
        all_filaments.sort(key=lambda f: f.get("remaining_weight") or 0)
    else:
        all_filaments.sort(key=sort_key)
    all_families_in_page = set()

    for spool in all_filaments:
        filament = spool.get("filament", {})
        hexes = []

        if filament.get("multi_color_hexes"):
            if isinstance(filament["multi_color_hexes"], str):
                hexes = filament["multi_color_hexes"].split(",")
            elif isinstance(filament["multi_color_hexes"], list):
                hexes = filament["multi_color_hexes"]
        elif filament.get("color_hex"):
            hexes = [filament["color_hex"]]

    
        families = set()
        for hx in hexes:
            fams = two_closest_families(hx, threshold=60)
            families.update(fams)
    
        spool["color_families"] = sorted(families)
        all_families_in_page.update(families)
    selected_family = request.args.get("color")
    if selected_family:
        all_filaments = [
            f for f in all_filaments
            if selected_family in f.get("color_families", [])
        ]
    total = len(all_filaments)
    total_pages = math.ceil(total / per_page)
    filaments_page = all_filaments[(page-1)*per_page : page*per_page]
    assign_print_id = request.args.get("assign_print_id")
    assign_filament_index = request.args.get("assign_filament_index")
    assign_page = request.args.get("assign_page")
    assign_search = request.args.get("assign_search")
    is_assign_mode = all([assign_print_id, assign_filament_index])
    return render_template(
        "fill.html",
        filaments=filaments_page,
        page=page,
        total_pages=total_pages,
        search=search,
        sort=sort,
        all_families=sorted(all_families_in_page),
        selected_family=selected_family,
        page_title="Fill",
        ams_id=ams_id, 
        tray_id=tray_id,
        assign_print_id=assign_print_id,
        assign_filament_index=assign_filament_index,
        assign_page=assign_page,
        assign_search=assign_search,
        is_assign_mode=is_assign_mode
    )

@app.route("/spool_info")
def spool_info():
  if not isMqttClientConnected():
    return render_template('error.html', exception="MQTT is disconnected. Is the printer online?")
    
  try:
    tag_id = request.args.get("tag_id", "-1")
    spool_id = request.args.get("spool_id", -1)
    last_ams_config = getLastAMSConfig()
    ams_data = last_ams_config.get("ams", [])
    vt_tray_data = last_ams_config.get("vt_tray", {})
    spool_list = fetchSpools()
    
    issue = False
    #TODO: Fix issue when external spool info is reset via bambulab interface
    augmentTrayDataWithSpoolMan(spool_list, vt_tray_data, trayUid(EXTERNAL_SPOOL_AMS_ID, EXTERNAL_SPOOL_ID))
    issue |= vt_tray_data["issue"]

    for ams in ams_data:
      for tray in ams["tray"]:
        augmentTrayDataWithSpoolMan(spool_list, tray, trayUid(ams["id"], tray["id"]))
        issue |= tray["issue"]

    if not tag_id:
      return render_template('error.html', exception="TAG ID is required as a query parameter (e.g., ?tag_id=RFID123)")

    spools = fetchSpools()
    current_spool = None
    for spool in spools:
      if spool['id'] == int(spool_id):
        current_spool = spool
        break

      if not spool.get("extra", {}).get("tag"):
        continue

      tag = json.loads(spool["extra"]["tag"])
      if tag != tag_id:
        continue

      current_spool = spool

    if current_spool:
      # TODO: missing current_spool
      return render_template('spool_info.html', tag_id=tag_id, current_spool=current_spool, ams_data=ams_data, vt_tray_data=vt_tray_data)
    else:
      return render_template('error.html', exception="Spool not found")
  except Exception as e:
    traceback.print_exc()
    return render_template('error.html', exception=str(e))


@app.route("/tray_load")
def tray_load():
  if not isMqttClientConnected():
    return render_template('error.html', exception="MQTT is disconnected. Is the printer online?")
  
  tag_id = request.args.get("tag_id")
  ams_id = request.args.get("ams")
  tray_id = request.args.get("tray")
  spool_id = request.args.get("spool_id")

  if not all([ams_id, tray_id, spool_id]):
    return render_template('error.html', exception="Missing AMS ID, or Tray ID or spool_id.")

  try:
    # Update Spoolman with the selected tray
    spool_data = getSpoolById(spool_id)
    setActiveTray(spool_id, spool_data["extra"], ams_id, tray_id)
    setActiveSpool(ams_id, tray_id, spool_data)

    return redirect(url_for('home', success_message=f"Updated Spool ID {spool_id} with TAG id {tag_id} to AMS {ams_id}, Tray {tray_id}."))
  except Exception as e:
    traceback.print_exc()
    return render_template('error.html', exception=str(e))

def setActiveSpool(ams_id, tray_id, spool_data):
  if not isMqttClientConnected():
    return render_template('error.html', exception="MQTT is disconnected. Is the printer online?")
  
  ams_message = AMS_FILAMENT_SETTING
  ams_message["print"]["sequence_id"] = 0
  ams_message["print"]["ams_id"] = int(ams_id)
  ams_message["print"]["tray_id"] = int(tray_id)
  
  if "color_hex" in spool_data["filament"]:
    ams_message["print"]["tray_color"] = spool_data["filament"]["color_hex"].upper() + "FF"
  else:
    ams_message["print"]["tray_color"] = spool_data["filament"]["multi_color_hexes"].split(',')[0].upper() + "FF"
      
  if "nozzle_temperature" in spool_data["filament"]["extra"]:
    nozzle_temperature_range = spool_data["filament"]["extra"]["nozzle_temperature"].strip("[]").split(",")
    ams_message["print"]["nozzle_temp_min"] = int(nozzle_temperature_range[0])
    ams_message["print"]["nozzle_temp_max"] = int(nozzle_temperature_range[1])
  else:
    nozzle_temperature_range_obj = generate_filament_temperatures(spool_data["filament"]["material"],
                                                                  spool_data["filament"]["vendor"]["name"])
    ams_message["print"]["nozzle_temp_min"] = int(nozzle_temperature_range_obj["filament_min_temp"])
    ams_message["print"]["nozzle_temp_max"] = int(nozzle_temperature_range_obj["filament_max_temp"])

  ams_message["print"]["tray_type"] = spool_data["filament"]["material"]

  filament_brand_code = {}
  filament_brand_code["brand_code"] = spool_data["filament"]["extra"].get("filament_id", "").strip('"')
  filament_brand_code["sub_brand_code"] = ""

  if filament_brand_code["brand_code"] == "":
    filament_brand_code = generate_filament_brand_code(spool_data["filament"]["material"],
                                                      spool_data["filament"]["vendor"]["name"],
                                                      spool_data["filament"]["extra"].get("type", ""))
    
  ams_message["print"]["tray_info_idx"] = filament_brand_code["brand_code"]

  # TODO: test sub_brand_code
  # ams_message["print"]["tray_sub_brands"] = filament_brand_code["sub_brand_code"]
  ams_message["print"]["tray_sub_brands"] = ""

  print(ams_message)
  #publish(getMqttClient(), ams_message)
@app.route("/")
def home():
  if not isMqttClientConnected():
    return render_template('error.html', exception="MQTT is disconnected. Is the printer online?")
    
  try:
    last_ams_config = getLastAMSConfig()
    ams_data = last_ams_config.get("ams", [])
    vt_tray_data = last_ams_config.get("vt_tray", {})
    spool_list = fetchSpools()
    success_message = request.args.get("success_message")
    
    issue = False
    #TODO: Fix issue when external spool info is reset via bambulab interface
    augmentTrayDataWithSpoolMan(spool_list, vt_tray_data, trayUid(EXTERNAL_SPOOL_AMS_ID, EXTERNAL_SPOOL_ID))
    issue |= vt_tray_data["issue"]

    for ams in ams_data:
      for tray in ams["tray"]:
        augmentTrayDataWithSpoolMan(spool_list, tray, trayUid(ams["id"], tray["id"]))
        issue |= tray["issue"]
      location = ''
      if LOCATION_MAPPING != '' :
        d = dict(item.split(":", 1) for item in LOCATION_MAPPING.split(";"))
        ams_name='AMS_'+str(ams["id"])
        if ams_name in d:
            location = d[ams_name]
      ams['location']=location
    if AMS_ORDER != '':
      mapping = {int(k): int(v) for k, v in (item.split(":") for item in AMS_ORDER.split(";"))}
      reordered = [None] * len(ams_data)
      for src_index, dst_index in mapping.items():
          reordered[dst_index] = ams_data[src_index]
      ams_data=reordered

    # Nouveau : si ?webview=1 → on met le cookie
    resp = make_response(render_template(
        'index.html',
        success_message=success_message,
        ams_data=ams_data,
        vt_tray_data=vt_tray_data,
        issue=issue,
        page_title="Home"
    ))

    if request.args.get("webview") == "1":
        resp.set_cookie("webview", "1")

    return resp
  except Exception as e:
    traceback.print_exc()
    return render_template('error.html', exception=str(e))

def sort_spools(spools):
  def condition(item):
    # Ensure the item has an "extra" key and is a dictionary
    if not isinstance(item, dict) or "extra" not in item or not isinstance(item["extra"], dict):
      return False

    # Check the specified condition
    return item["extra"].get("tag") or item["extra"].get("tag") == ""

  # Sort with the custom condition: False values come first
  return sorted(spools, key=lambda spool: bool(condition(spool)))

@app.route("/assign_tag")
def assign_tag():
  if not isMqttClientConnected():
    return render_template('error.html', exception="MQTT is disconnected. Is the printer online?")
    
  try:
    spools = sort_spools(fetchSpools())

    return render_template('assign_tag.html', spools=spools,
        page_title="NFC")
  except Exception as e:
    traceback.print_exc()
    return render_template('error.html', exception=str(e))

@app.route("/write_tag")
def write_tag():
  try:
    spool_id = request.args.get("spool_id")

    if not spool_id:
      return render_template('error.html', exception="spool ID is required as a query parameter (e.g., ?spool_id=1)")

    myuuid = str(uuid.uuid4())

    patchExtraTags(spool_id, {}, {
      "tag": json.dumps(myuuid),
    })
    return render_template('write_tag.html', myuuid=myuuid)
  except Exception as e:
    traceback.print_exc()
    return render_template('error.html', exception=str(e))

@app.route('/', methods=['GET'])
def health():
  return "OK", 200

@app.route("/print_history")
def print_history():
    spoolman_settings = getSettings()

    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))

    filters = {
        "filament_type": request.args.getlist("filament_type"),
        "color": request.args.getlist("color"),
        "filament_id": request.args.getlist("filament_id"),
        "status": request.args.getlist("status")
    }
    filters["filament_id"] = [v for v in filters["filament_id"] if v.strip()]
    search = request.args.get("search", "").strip()

    focus_print_id = request.args.get("focus_print_id", type=int)
    focus_group_id = request.args.get("focus_group_id", type=int)

    raw_prints = get_prints_with_filament(filters=filters, search=search)
    spool_list = fetchSpools(False, True)
    entries = {}

    for p in raw_prints:
        if p["duration"] is None:
            p["duration"] = 0
        p["duration"] /= 3600
        p["electric_cost"] = p["duration"] * float(COST_BY_HOUR)
        p["filament_usage"] = json.loads(p["filament_info"])
        p["total_cost"] = 0
        p["total_normal_cost"] = 0
        p["tags"] = get_tags_for_print(p["id"])
        p["total_weight"] = sum(f["grams_used"] for f in p["filament_usage"])
        p["translated_name"] = p.get("translated_name", "")
        p["total_price"] = p.get("sold_price_total", 0)

        for filament in p["filament_usage"]:
            if filament["spool_id"]:
                for spool in spool_list:
                    if spool['id'] == filament["spool_id"]:
                        filament["spool"] = spool
                        filament["cost"] = filament['grams_used'] * spool.get('cost_per_gram', 20.0)
                        filament["normal_cost"] = filament['grams_used'] * spool.get('filament_cost_per_gram', 20.0)
                        p["total_cost"] += filament["cost"]
                        p["total_normal_cost"] += filament["normal_cost"]
                        break
            filament.setdefault("cost", 20.0)

        p["full_cost"] = p["total_cost"] + p["electric_cost"]
        p["full_normal_cost"] = p["total_normal_cost"] + p["electric_cost"]

        if not p.get("number_of_items"):
            p["number_of_items"] = 1
        p["full_cost_by_item"] = p["full_cost"] / p["number_of_items"]
        p["full_normal_cost_by_item"] = p["full_normal_cost"] / p["number_of_items"]
        if p.get("image_file") and p["image_file"].endswith(".png"):
            model_file = p["image_file"].replace(".png", ".3mf")
            model_path = os.path.join(app.static_folder, 'prints', model_file)
            p["model_file"] = model_file if os.path.isfile(model_path) else None
        else:
            p["model_file"] = None

        if p.get("group_id"):
            gid = p["group_id"]
            entry_key = f"group_{gid}"
            entry = entries.get(entry_key)
            if not entry or entry.get("type") != "group":
                entry = {
                    "type": "group",
                    "id": gid,
                    "name": p.get("group_name", f"Groupe {gid}"),
                    "prints": [],
                    "total_duration": 0,
                    "total_cost": 0,
                    "total_normal_cost": 0,
                    "total_weight": 0,
                    "max_id": 0,
                    "latest_date": p["print_date"],
                    "thumbnail": None,  # sera défini plus bas
                    "filament_usage": {},
                    "number_of_items": p.get("group_number_of_items") or 1,
                    "primary_print_id": p.get("group_primary_print_id"),
                }
                entries[entry_key] = entry
        
            entry["prints"].append(p)
            entry["total_duration"] += p["duration"]
            entry["total_cost"] += p["full_cost"]
            entry["total_normal_cost"] += p["full_normal_cost"]
            entry["total_weight"] += p["total_weight"]
            entry["total_price"] = p.get("group_sold_price_total", 0)
            entry["sold_units"] = p.get("group_sold_units", 0)
        
            if p["id"] > entry["max_id"]:
                entry["max_id"] = p["id"]
                entry["latest_date"] = p["print_date"]
        
            # Définir la miniature en fonction du primary_print_id s’il est défini
            if entry.get("primary_print_id"):
                if p["id"] == entry["primary_print_id"]:
                    entry["thumbnail"] = p["image_file"]
            elif not entry.get("thumbnail"):
                entry["thumbnail"] = p["image_file"]
        
            for filament in p["filament_usage"]:
                key = filament["spool_id"] or f"{filament['filament_type']}-{filament['color']}"
                if key not in entry["filament_usage"]:
                    entry["filament_usage"][key] = {
                        "grams_used": filament["grams_used"],
                        "cost": filament.get("cost", 20.0),
                        "normal_cost": filament.get("normal_cost", 20.0),
                        "spool": filament.get("spool"),
                        "spool_id": filament.get("spool_id"),
                        "filament_type": filament.get("filament_type"),
                        "color": filament.get("color")
                    }
                else:
                    usage = entry["filament_usage"][key]
                    usage["grams_used"] += filament["grams_used"]
                    usage["cost"] += filament.get("cost", 20.0)
                    usage["normal_cost"] += filament.get("normal_cost", 20.0)
        else:
            entries[f"print_{p['id']}"] = {
                "type": "single",
                "print": p,
                "max_id": p["id"]
            }

    for entry in entries.values():
        if entry["type"] == "group":
            entry["total_electric_cost"] = entry["total_duration"] * float(COST_BY_HOUR)
            entry["total_filament_cost"] = entry["total_cost"] - entry["total_electric_cost"]
            entry["total_filament_normal_cost"] = entry["total_normal_cost"] - entry["total_electric_cost"]
            entry["full_cost_by_item"] = entry["total_cost"] / entry["number_of_items"]
            entry["full_normal_cost_by_item"] = entry["total_normal_cost"] / entry["number_of_items"]

    entries_list = sorted(entries.values(), key=lambda e: e["max_id"], reverse=True)
    total_pages = (len(entries_list) + per_page - 1) // per_page
    paged_entries = entries_list[(page - 1) * per_page : page * per_page]

    if focus_print_id and not focus_group_id:
        for entry in entries_list:
            if entry["type"] == "single" and entry["print"]["id"] == focus_print_id:
                if entry["print"].get("group_id"):
                    focus_group_id = entry["print"]["group_id"]
                break
            elif entry["type"] == "group":
                for p in entry["prints"]:
                    if p["id"] == focus_print_id:
                        focus_group_id = p.get("group_id")
                        break
                if focus_group_id:
                    break

    distinct_values = get_distinct_values()
    args = request.args.to_dict(flat=False)
    args.pop('page', None)
    groups_list = get_print_groups()
    pagination_pages = compute_pagination_pages(page, total_pages)

    filters["filament_id"] = [fid for group in filters["filament_id"] for fid in group.split(',') if fid]
    status_values = sorted(set(p.get("status") for p in raw_prints if p.get("status")))
    return render_template(
        'print_history.html',
        entries=paged_entries,
        groups_list=groups_list,
        currencysymbol=spoolman_settings["currency_symbol"],
        page=page,
        total_pages=total_pages,
        filters=filters,
        distinct_values=distinct_values,
        args=args,
        search=search,
        pagination_pages=pagination_pages,
        focus_print_id=focus_print_id,
        focus_group_id=focus_group_id,
        status_values=status_values,
        page_title="History"
    )

@app.route("/print_select_spool")
def print_select_spool():

  try:
    ams_slot = request.args.get("ams_slot")
    print_id = request.args.get("print_id")

    if not all([ams_slot, print_id]):
      return render_template('error.html', exception="Missing spool ID or print ID.")

    spools = fetchSpools()
        
    return render_template('print_select_spool.html', spools=spools, ams_slot=ams_slot, print_id=print_id)
  except Exception as e:
    traceback.print_exc()
    return render_template('error.html', exception=str(e))

@app.route("/history/delete/<int:print_id>", methods=["POST"])
def delete_print_history(print_id):
    data = request.get_json()
    restock = data.get("restock", False)
    ratios = data.get("ratios", {})  # dict {spool_id: ratio_en_%}

    # Récupère la liste des consommations de filament pour ce print
    filament_usages = get_filament_for_print(print_id)

    if restock:
        for usage in filament_usages:
            spool_id = usage["spool_id"]
            grams_used = usage["grams_used"]

            if spool_id:
                # Si un ratio est fourni pour ce spool_id, sinon 100%
                ratio_percent = ratios.get(str(spool_id), 100)
                ratio = max(0, min(100, ratio_percent)) / 100.0
                adjusted_grams = grams_used * ratio

                # Remet en stock la quantité ajustée
                consumeSpool(spool_id, -adjusted_grams)

    # Supprime le print et ses usages
    delete_print(print_id)

    return jsonify({"status": "ok"})

@app.route("/history/reajust/<int:print_id>", methods=["POST"])
def reajust_print_history(print_id):
    data = request.get_json()
    ratios = data.get("ratios", {})

    filament_usages = get_filament_for_print(print_id)

    for usage in filament_usages:
        spool_id = usage["spool_id"]
        grams_used = usage["grams_used"]

        if spool_id:
            ratio_percent = ratios.get(str(spool_id), 100)
            ratio = max(0, min(100, ratio_percent)) / 100.0
            adjusted_grams = grams_used * ratio

            consumeSpool(spool_id, -adjusted_grams)

            # 🔷 Met à jour filament_usage
            new_grams_used = grams_used - adjusted_grams
            update_filament_usage(print_id, spool_id, new_grams_used)

    return jsonify({"status": "ok"})

@app.route("/history/<int:print_id>/filaments", methods=["GET"])
def get_print_filaments(print_id):
    filament_usages = get_filament_for_print(print_id)

    enriched = []
    for usage in filament_usages:
        spool_id = usage["spool_id"]
        grams_used = usage["grams_used"]

        spool = getSpoolById(spool_id) if spool_id else None
        if spool:
            color = spool.get("filament", {}).get("color_hex")
            vendor = spool.get("filament", {}).get("vendor", {}).get("name", "UnknownVendor")
            material = spool.get("filament", {}).get("material", "UnknownMaterial")
            realName = spool.get("filament", {}).get("name", "UnknownName")
            name = f"#{spool_id} - {realName} - {vendor} - {material}"
        else:
            color = usage.get("color", "#000000")
            name = usage.get("filament_type", "N/A")
        if color:
            color = f"#{color}"
        enriched.append({
            "spool_id": spool_id,
            "grams_used": grams_used,
            "name": name,
            "color": color
        })

    return jsonify(enriched)

@app.route("/history/<int:print_id>/tags", methods=["GET"])
def get_tags(print_id):
    tags = get_tags_for_print(print_id)
    return jsonify(tags)

@app.route("/history/<int:print_id>/tags/add", methods=["POST"])
def add_tag(print_id):
    tag = request.form.get("tag", "")
    # On découpe la chaîne en mots
    tags = [word.strip() for word in tag.split() if word.strip()]
    for t in tags:
        add_tag_to_print(print_id, t)
    return jsonify({"status": "ok"})

@app.route("/history/<int:print_id>/tags/remove", methods=["POST"])
def remove_tag(print_id):
    tag = request.form.get("tag")
    if tag:
        remove_tag_from_print(print_id, tag)
    return jsonify({"status": "ok"})
    
@app.route("/filaments")
def filaments():
    page = int(request.args.get("page", 1))
    per_page = 25
    search = request.args.get("search", "").lower()
    sort = request.args.get("sort", "default")

    all_filaments = fetchSpools() or []

    # filtre nom / couleur
    if search:
        search_terms = search.split()

        def matches(f):
            filament = f.get("filament", {})
            vendor = filament.get("vendor", {})
            fields = [
                filament.get("name", "").lower(),
                filament.get("material", "").lower(),
                vendor.get("name", "").lower(),
                f.get("location", "").lower(),
            ]
            return all(
                any(term in field for field in fields)
                for term in search_terms
            )

        all_filaments = [f for f in all_filaments if matches(f)]

    def sort_key(f):
        filament = f.get("filament", {})
        vendor = filament.get("vendor", {})
        return (
            f.get("location", "").lower(),
            filament.get("material", "").lower(),
            vendor.get("name", "").lower(),
            filament.get("name", "").lower()
        )

    if sort == "remaining":
        all_filaments.sort(key=lambda f: f.get("remaining_weight") or 0)
    else:
        all_filaments.sort(key=sort_key)

    all_families_in_page = set()

    for spool in all_filaments:
        filament = spool.get("filament", {})
        hexes = []

        if filament.get("multi_color_hexes"):
            if isinstance(filament["multi_color_hexes"], str):
                hexes = filament["multi_color_hexes"].split(",")
            elif isinstance(filament["multi_color_hexes"], list):
                hexes = filament["multi_color_hexes"]
        elif filament.get("color_hex"):
            hexes = [filament["color_hex"]]

        families = set()
        for hx in hexes:
            fams = two_closest_families(hx, threshold=60)
            families.update(fams)

        spool["color_families"] = sorted(families)
        all_families_in_page.update(families)

    selected_family = request.args.get("color")
    if selected_family:
        all_filaments = [
            f for f in all_filaments
            if selected_family in f.get("color_families", [])
        ]

    total = len(all_filaments)
    total_pages = math.ceil(total / per_page)
    filaments_page = all_filaments[(page-1)*per_page : page*per_page]

    # ==== Statistiques ====
    vendor_names = {
        f.get("filament", {}).get("vendor", {}).get("name", "")
        for f in all_filaments
        if f.get("filament", {}).get("vendor")
    }
    total_vendors = len(vendor_names)
    total_remaining = sum(f.get("remaining_weight") or 0 for f in all_filaments)

    return render_template(
        "filaments.html",
        filaments=filaments_page,
        page=page,
        total_pages=total_pages,
        search=search,
        sort=sort,
        all_families=sorted(all_families_in_page),
        selected_family=selected_family,
        page_title="Filaments",
        total_filaments=total,
        total_vendors=total_vendors,
        total_remaining=total_remaining
    )

@app.route("/edit_print_name", methods=["POST"])
def edit_print_name():
    print_id = int(request.form["print_id"])
    new_filename = request.form.get("file_name", "").strip()
    page = int(request.form.get("page", 1))

    if new_filename:
        update_print_filename(print_id, new_filename)
    return redirect(url_for("print_history", page=page, search=request.args.get("search",''), focus_print_id=print_id))

    
@app.route("/edit_print_items", methods=["POST"])
def edit_print_items():
    print_id = int(request.form["print_id"])
    try:
        number_of_items = int(request.form["number_of_items"])
        if number_of_items < 1:
            number_of_items = 1
    except (ValueError, TypeError):
        number_of_items = 1

    page = int(request.form.get("page", 1))

    update_print_history_field(print_id, "number_of_items", number_of_items)

    return redirect(url_for("print_history", page=page, search=request.args.get("search",''), focus_print_id=print_id))

@app.route("/create_group", methods=["POST"])
def create_group():
    print_id = int(request.form["print_id"])
    group_name = request.form["group_name"].strip()
    page = int(request.form.get("page", 1))

    if group_name:
        group_id = create_print_group(group_name)
        update_print_history_field(print_id, "group_id", group_id)
        update_group_created_at(group_id)  # 🔷 Ajout ici
        return redirect(url_for("print_history", page=page, search=request.args.get("search",''), focus_group_id=group_id))

    return redirect(url_for("print_history", page=page, search=request.args.get("search",''), focus_print_id=print_id))
    
@app.route("/assign_to_group", methods=["POST"])
def assign_to_group():
    group_id_or_name = request.form["group_id_or_name"]
    print_id = int(request.form["print_id"])
    page = int(request.form.get("page", 1))

    if group_id_or_name.isdigit():
        group_id = int(group_id_or_name)
    else:
        group_id = create_print_group(group_id_or_name)

    update_print_history_field(print_id, "group_id", group_id)
    update_group_created_at(group_id)

    return redirect(url_for("print_history", page=page, search=request.args.get("search",''), focus_group_id=group_id))

@app.route("/remove_from_group", methods=["POST"])
def remove_from_group():
    print_id = int(request.form["print_id"])
    page = int(request.form.get("page", 1))

    group_id = get_group_id_of_print(print_id)

    update_print_history_field(print_id, "group_id", None)
    if group_id:
        update_group_created_at(group_id)

    return redirect(url_for("print_history", page=page, search=request.args.get("search",''), focus_print_id=print_id))

@app.route("/rename_group", methods=["POST"])
def rename_group():
    group_id = int(request.form["group_id"])
    group_name = request.form["group_name"].strip()
    page = int(request.form.get("page", 1))

    if group_name:
        update_print_group_field(group_id, "name", group_name)

    return redirect(url_for("print_history", page=page, search=request.args.get("search",''), focus_group_id=group_id))

@app.route("/edit_group_items", methods=["POST"])
def edit_group_items():
    group_id = int(request.form["group_id"])
    try:
        number_of_items = int(request.form["number_of_items"])
        if number_of_items < 1:
            number_of_items = 1
    except (ValueError, TypeError):
        number_of_items = 1

    page = int(request.form.get("page", 1))

    update_print_group_field(group_id, "number_of_items", number_of_items)

    return redirect(url_for("print_history", page=page, focus_group_id=group_id))

from datetime import datetime

@app.route("/api/groups/search")
def api_groups_search():
    q = request.args.get("q", "").strip()
    groups = get_print_groups()
    results = []
    for group in groups:
        if q.lower() in group["name"].lower():
            created_at_str = group.get("created_at")
            try:
                created_at = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
                created_at_fmt = created_at.strftime('%d/%m/%Y %H:%M')
            except Exception:
                created_at_fmt = created_at_str or "?"
            label = f"{group['name']} (créé le {created_at_fmt})"
            results.append({"id": group["id"], "text": label})
    return jsonify({"results": results})
    
@app.route('/spool/<int:spool_id>/reajust', methods=['POST'])
def reajust_spool_route(spool_id):
    try:
        new_weight = float(request.form.get('new_weight'))
    except (ValueError, TypeError):
        flash('Poids invalide', 'danger')
        return redirect(request.referrer or url_for('filament_page'))

    response = reajust_spool(spool_id, new_weight)
    return redirect(request.referrer or url_for('filament_page'))

@app.route('/spool/<int:spool_id>/archive', methods=['POST'])
def archive_spool_route(spool_id):
    response = archive_spool(spool_id)
    return redirect(request.referrer or url_for('filament_page'))

@app.route("/download_model/<filename>")
def download_model(filename):
    # Sécuriser le nom du fichier réel (dans 'static/prints')
    safe_filename = os.path.basename(filename)

    # Nom souhaité côté utilisateur (dans l'URL `?as=...`)
    requested_name = request.args.get("as", safe_filename)

    # Nettoyage du nom pour éviter les injections ou caractères suspects
    cleaned_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', requested_name)
    if not cleaned_name.endswith('.3mf'):
        cleaned_name += '.3mf'

    # Vérifie que le fichier existe avant d’envoyer
    full_path = os.path.join('static', 'prints', safe_filename)
    if not os.path.isfile(full_path):
        return abort(404)

    return send_from_directory(
        os.path.join('static', 'prints'),
        safe_filename,
        as_attachment=True,
        download_name=cleaned_name
    )

@app.route("/stats")
def stats():
    from print_history import get_distinct_values

    spoolman_settings = getSettings()

    # Récupérer les filtres, recherche et période
    filters = {
        "filament_type": request.args.getlist("filament_type"),
        "color": request.args.getlist("color")
    }
    search = request.args.get("search", "").strip()
    period = request.args.get("period", "all")

    # Calcul des stats avec filtres
    stats_data = get_statistics(period=period, filters=filters, search=search)

    return render_template(
        "stats.html",
        stats=stats_data,
        currencysymbol=spoolman_settings["currency_symbol"],
        selected_period=period,
        filters=filters,
        search=search,
        distinct_values=get_distinct_values(),
        page_title="Statistiques"
    )

@app.route("/adjust_duration", methods=["POST"])
def adjust_duration():
    print_id = int(request.form["print_id"])
    try:
        hours = float(request.form.get("hours", 0) or 0)
        minutes = float(request.form.get("minutes", 0) or 0)
    except ValueError:
        pass

    total_seconds = int((hours * 60 + minutes) * 60)

    try:
        adjustDuration(print_id, total_seconds)
    except Exception as e:
        pass

    return redirect(url_for("print_history", page=request.args.get("page"), search=request.args.get("search",''),focus_print_id=print_id))

@app.route("/set_group_primary", methods=["POST"])
def set_group_primary():
    print_id = int(request.form["print_id"])
    group_id = int(request.form["group_id"])
    set_group_primary_print(group_id, print_id)
    page = request.form.get("page", 1)
    search = request.form.get("search", "")
    return redirect(url_for("print_history", page=page, search=search, focus_group_id=group_id))

@app.route('/assign_spool_to_print', methods=['POST'])
def assign_spool_to_print():
    spool_id = int(request.form['spool_id'])
    print_id = int(request.form['print_id'])
    filament_index = int(request.form['filament_index'])  # correspond à ams_slot dans ta BDD
    page = request.form.get('page', 1)
    search = request.form.get('search', '')

    update_filament_spool(print_id=print_id, filament_id=filament_index, spool_id=spool_id)

    return redirect(url_for('print_history', page=page, focus_print_id=print_id, search=search))

@app.route("/change_print_status", methods=["POST"])
def change_print_status():
    print_id = int(request.form.get("print_id"))
    new_status = request.form.get("status", "SUCCESS").strip()
    note = request.form.get("status_note", "").strip()
    page = request.form.get("page", 1)
    search = request.form.get("search", "")

    if new_status not in {"SUCCESS", "IN_PROGRESS", "FAILED", "PARTIAL", "TO_REDO"}:
        return redirect(url_for("print_history", page=page, search=search, focus_print_id=print_id))

    update_print_history_field(print_id, "status", new_status)
    update_print_history_field(print_id, "status_note", note)

    return redirect(url_for("print_history", page=page, search=search, focus_print_id=print_id))

@app.route("/set_sold_price", methods=["POST"])
def set_sold_price():
    try:
        item_id = int(request.form.get("id"))
        is_group = bool(request.form.get("is_group"))
        total_price = float(request.form.get("total_price") or 0)
        sold_units = int(request.form.get("sold_units") or 0)

        if item_id <= 0 or total_price < 0 or sold_units < 0:
            return redirect(request.referrer or url_for("print_history"))

        # Appliquer les changements
        set_sold_info(print_id=item_id, is_group=is_group, total_price=total_price, sold_units=sold_units)

        # Générer dynamiquement les paramètres à préserver
        preserved_args = {
            key: request.form.getlist(key)
            for key in request.form
            if key not in {"id", "is_group", "total_price", "sold_units"}
        }

        return redirect(url_for("print_history", **preserved_args, focus_group_id=item_id if is_group else None, focus_print_id=item_id if not is_group else None))

    except Exception as e:
        return redirect(request.referrer or url_for("print_history"))

app.register_blueprint(auth_bp)