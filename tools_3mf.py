import requests
import zipfile
import tempfile
import xml.etree.ElementTree as ET
import pycurl
import urllib.parse
import os
import shutil
import re
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse, unquote
from config import get_app_setting
import logging
import re

logger = logging.getLogger(__name__)


def parse_ftp_listing(line):
    """Parse a line from an FTP LIST command."""
    parts = line.split(maxsplit=8)
    if len(parts) < 9:
        return None
    return {
        'permissions': parts[0],
        'links': int(parts[1]),
        'owner': parts[2],
        'group': parts[3],
        'size': int(parts[4]),
        'month': parts[5],
        'day': int(parts[6]),
        'time_or_year': parts[7],
        'name': parts[8]
    }

def get_base_name(filename):
    return filename.rsplit('.', 1)[0]

def parse_date(item):
    """Parse the date and time from the FTP listing item."""
    try:
        date_str = f"{item['month']} {item['day']} {item['time_or_year']}"
        return datetime.strptime(date_str, "%b %d %H:%M")
    except ValueError:
        return None

def get_filament_order(file):
    filament_order = {} 
    new_color_count = 0 

    for line in file:
        match_filament = re.match(r"^M620 S(\d+)[^;\r\n]*$", line.decode("utf-8").strip())
        if match_filament:
            filament = int(match_filament.group(1))
            if filament not in filament_order and int(filament) != 255:
                filament_order[int(filament)] = new_color_count
                new_color_count += 1

    if len(filament_order) == 0:
       filament_order = {1:0}

    return filament_order

def download3mfFromCloud(url, destFile):
  logger.info("Downloading 3MF file from cloud...")
  # Download the file and save it to the temporary file
  response = requests.get(url,timeout=20)
  response.raise_for_status()
  destFile.write(response.content)

def encode_custom_hex(filename):
    return ''.join(f"{ord(c):02x}" if c in "/:*" else c for c in filename)

def download3mfFromFTP(filename, taskname, destFile):
    """
    Mono-fonction FTPS:
      - attend l’apparition du fichier,
      - vérifie que MDTM (UTC) est à moins de 60s de maintenant,
      - vérifie la stabilité (taille + mtime) sur 3 cycles,
      - télécharge,
      - timeout global: 240s.
    """
    CHECK_INTERVAL   = 5           # secondes entre sondes
    TIMEOUT_SEC      = 240         # timeout global
    STABLE_CYCLES    = 3           # cycles identiques requis
    FRESH_MAX_AGE_S  = 60          # fichier "frais" = mtime <= 60s
    MDTM_RE          = re.compile(r"(?:^|[\r\n])\s*\d{3}\s+(\d{14})(?:\.\d+)?\b", re.IGNORECASE)
    BACKOFF_ON_SIZE_CHANGE = 0.25  # s — backoff avant MDTM si la taille vient de changer
    RECREATE_AFTER_STALE   = 2     # cycles consécutifs avec size_ok && mtime=None -> recreate session

    ftp_host = get_app_setting("PRINTER_IP","")
    ftp_user = "bblp"
    ftp_pass = get_app_setting("PRINTER_ACCESS_CODE","")

    logger.info("Downloading 3MF file from FTP...")
    taskname = encode_custom_hex(taskname)

    remote_dir  = "/cache"
    remote_name = f"{taskname}.gcode.3mf"
    remote_name_enc = urllib.parse.quote(remote_name)
    url       = f"ftps://{ftp_host}{remote_dir}/{remote_name_enc}"
    raw_path  = f"{remote_dir}/{remote_name}"          # pour QUOTE MDTM (non encodé)
    local_path = destFile.name

    logger.debug(f"Waiting for file to appear and stabilize: {url}")

    start = time.time()
    last_sig = None          # (size, mtime) avec -1 si None
    last_size = None         # pour détecter un changement de taille
    stable = 0
    size_ok_mtime_none_streak = 0

    def _init_curl():
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(c.USERPWD, f"{ftp_user}:{ftp_pass}")
        c.setopt(c.SSL_VERIFYPEER, 0)
        c.setopt(c.SSL_VERIFYHOST, 0)
        c.setopt(c.FTP_SSL, c.FTPSSL_ALL)
        c.setopt(c.FTPSSLAUTH, c.FTPAUTH_TLS)
        c.setopt(c.FTP_FILEMETHOD, c.FTPMETHOD_NOCWD)
        c.setopt(c.TRANSFERTEXT, False)      # binaire
        c.setopt(c.CONNECTTIMEOUT, 10)
        c.setopt(c.TIMEOUT, CHECK_INTERVAL)
        c.setopt(c.NOBODY, True)
        return c

    c = _init_curl()
    try:
        while time.time() - start < TIMEOUT_SEC:
            size = None
            mtime = None

            # 1) Taille (HEAD/SIZE via NOBODY)
            try:
                c.perform()
                cl = int(c.getinfo(c.CONTENT_LENGTH_DOWNLOAD))
                size = cl if cl >= 0 else None
            except pycurl.error:
                pass  # pas encore disponible

            # 2) MDTM (transcript)
            # Backoff si la taille vient de changer (fenêtre de course fs/rename)
            if last_size is not None and size is not None and size != last_size:
                logger.debug(f"⏳ Backoff {BACKOFF_ON_SIZE_CHANGE*1000:.0f}ms avant MDTM (changement de taille détecté)")
                time.sleep(BACKOFF_ON_SIZE_CHANGE)

            buf = []
            def _dbg(_t, m):
                try:
                    buf.append(m.decode("latin1", "ignore"))
                except Exception:
                    pass

            c.setopt(c.VERBOSE, True)
            c.setopt(c.DEBUGFUNCTION, _dbg)
            quote_cmd = f"MDTM {raw_path}".encode("utf-8")
            c.setopt(c.QUOTE, [quote_cmd])
            try:
                c.perform()
            except pycurl.error:
                pass
            finally:
                c.setopt(c.QUOTE, [])
                c.setopt(c.VERBOSE, False)
                c.setopt(c.DEBUGFUNCTION, (lambda *a, **k: None))

            transcript = "".join(buf)
            m = MDTM_RE.search(transcript)
            if m:
                try:
                    dt = datetime.strptime(m.group(1)[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                    mtime = int(dt.timestamp())
                except Exception:
                    mtime = None

            # 3) Fraîcheur / stabilité
            now_utc = int(datetime.now(timezone.utc).timestamp())
            age_s = (now_utc - mtime) if mtime is not None else None
            fresh_ok = (mtime is not None) and (0 <= age_s <= FRESH_MAX_AGE_S)
            size_ok  = (size is not None and size > 0)

            sig = (size if size is not None else -1, mtime if mtime is not None else -1)
            logger.debug(f"📏 size={size}  🕒 mtime={mtime}  ⏳age={str(age_s)+'s' if age_s is not None else '—'}  ✅fresh={fresh_ok}")

            # 4) Détection du cas "session sourde à MDTM" → recréer la connexion
            if size_ok and (mtime is None):
                size_ok_mtime_none_streak += 1
                if size_ok_mtime_none_streak >= RECREATE_AFTER_STALE:
                    logger.warning("🔌 Session FTPS possiblement 'sourde' à MDTM — recréation de la connexion…")
                    try:
                        c.close()
                    except Exception:
                        pass
                    c = _init_curl()
                    size_ok_mtime_none_streak = 0
                    last_sig = None
                    stable = 0
                    # petit délai pour laisser la session serveur se poser
                    time.sleep(0.2)
                    last_size = size  # mémorise le dernier size vu
                    continue
            else:
                size_ok_mtime_none_streak = 0

            # 5) Stabilisation
            if size_ok and fresh_ok:
                if sig == last_sig:
                    stable += 1
                    logger.debug(f"✅ Stable {stable}/{STABLE_CYCLES}")
                    if stable >= STABLE_CYCLES:
                        break
                else:
                    last_sig = sig
                    stable = 1
                    logger.debug("🔁 Changement détecté. Reset compteur -> 1")
            else:
                if sig != last_sig:
                    last_sig = sig
                    stable = 0
                    logger.debug("⏳ Pas prêt (taille/mtime). Reset compteur -> 0")

            last_size = size
            time.sleep(CHECK_INTERVAL)
        else:
            raise TimeoutError("Timeout 180s: fichier introuvable/non frais ou taille non stable.")

        # Téléchargement (même logique)
        logger.info("📥 Fichier stable. Lancement du download...")
        c.setopt(c.NOBODY, False)
        c.setopt(c.TIMEOUT, 0)  # pas de petit timeout pendant le download

        for attempt in range(1, 4):
            try:
                with open(local_path, "wb") as f:
                    c.setopt(c.WRITEDATA, f)
                    c.perform()
                logger.info(f"✅ Download OK -> {local_path} (~{size} bytes).")
                return local_path
            except pycurl.error as e:
                code = e.args[0] if e.args else None
                if code == 78 and attempt < 3:
                    logger.warning(f"⚠️ RETR 550 ? Retry {attempt}/2…")
                    c.setopt(c.FTP_FILEMETHOD, c.FTPMETHOD_SINGLECWD)
                    time.sleep(0.3 * attempt)
                    continue
                raise
    finally:
        c.close()


def download3mfFromLocalFilesystem(path, destFile):
  with open(path, "rb") as src_file:
    destFile.write(src_file.read())

def getMetaDataFrom3mf(url,taskname):
  """
  Download a 3MF file from a URL, unzip it, and parse filament usage.

  Args:
      url (str): URL to the 3MF file.

  Returns:
      list[dict]: List of dictionaries with `tray_info_idx` and `used_g`.
  """
  try:
    metadata = {}

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete_on_close=False,delete=True, suffix=".3mf") as temp_file:
      temp_file_name = temp_file.name
      
      if url.startswith("http"):
        download3mfFromCloud(url, temp_file)
        #download3mfFromFTP(url.replace("ftp://", ""), taskname, temp_file)
      elif url.startswith("local:"):
        download3mfFromLocalFilesystem(url.replace("local:", ""), temp_file)
      else:
        download3mfFromFTP(url.replace("ftp://", ""), taskname, temp_file)
      
      temp_file.close()

      parsed_url = urlparse(url)
      metadata["file"] = os.path.basename(parsed_url.path)

      logger.info(f"3MF file downloaded and saved as {temp_file_name}.")

      # Unzip the 3MF file
      with zipfile.ZipFile(temp_file_name, 'r') as z:
        metadata["title"] = ""
        model_path = "3D/3dmodel.model"
        if model_path in z.namelist():
            with z.open(model_path) as model_file:
                try:
                    model_tree = ET.parse(model_file)
                    model_root = model_tree.getroot()
                    for meta in model_root.findall(".//{*}metadata"):
                        if meta.attrib.get("name", "").lower() == "title":
                            metadata["title"] = meta.text.strip()
                            break
                except:
                    pass  # laisser title vide en cas d'erreur
        else:
            logger.info(f"Fichier '{model_path}' non trouvé dans l'archive.")
        # Check for the Metadata/slice_info.config file
        slice_info_path = "Metadata/slice_info.config"
        if slice_info_path in z.namelist():
          with z.open(slice_info_path) as slice_info_file:
            # Parse the XML content of the file
            tree = ET.parse(slice_info_file)
            root = tree.getroot()

            # Extract id and used_g from each filament
            """
            <?xml version="1.0" encoding="UTF-8"?>
            <config>
              <header>
                <header_item key="X-BBL-Client-Type" value="slicer"/>
                <header_item key="X-BBL-Client-Version" value="01.10.01.50"/>
              </header>
              <plate>
                <metadata key="index" value="1"/>
                <metadata key="printer_model_id" value="N2S"/>
                <metadata key="nozzle_diameters" value="0.4"/>
                <metadata key="timelapse_type" value="0"/>
                <metadata key="prediction" value="5450"/>
                <metadata key="weight" value="26.91"/>
                <metadata key="outside" value="false"/>
                <metadata key="support_used" value="false"/>
                <metadata key="label_object_enabled" value="true"/>
                <object identify_id="930" name="FILENAME.3mf" skipped="false" />
                <object identify_id="1030" name="FILENAME.3mf" skipped="false" />
                <object identify_id="1130" name="FILENAME.3mf" skipped="false" />
                <object identify_id="1230" name="FILENAME.3mf" skipped="false" />
                <object identify_id="1330" name="FILENAME.3mf" skipped="false" />
                <object identify_id="1430" name="FILENAME.3mf" skipped="false" />
                <object identify_id="1530" name="FILENAME.3mf" skipped="false" />
                <object identify_id="1630" name="FILENAME.3mf" skipped="false" />
                <object identify_id="1730" name="FILENAME.3mf" skipped="false" />
                <object identify_id="1830" name="FILENAME.3mf" skipped="false" />
                <object identify_id="1930" name="FILENAME.3mf" skipped="false" />
                <object identify_id="2030" name="FILENAME.3mf" skipped="false" />
                <object identify_id="2130" name="FILENAME.3mf" skipped="false" />
                <object identify_id="2230" name="FILENAME.3mf" skipped="false" />
                <filament id="1" tray_info_idx="GFL99" type="PLA" color="#0DFF00" used_m="6.79" used_g="20.26" />
                <filament id="2" tray_info_idx="GFL99" type="PLA" color="#000000" used_m="0.72" used_g="2.15" />
                <filament id="6" tray_info_idx="GFL99" type="PLA" color="#0DFF00" used_m="1.20" used_g="3.58" />
                <filament id="7" tray_info_idx="GFL99" type="PLA" color="#000000" used_m="0.31" used_g="0.92" />
                <warning msg="bed_temperature_too_high_than_filament" level="1" error_code ="1000C001"  />
              </plate>
            </config>
            """
            
            for meta in root.findall(".//plate/metadata"):
              if meta.attrib.get("key") == "index":
                  metadata["plateID"] = meta.attrib.get("value", "")
              if meta.attrib.get("key") == "prediction":
                  metadata["duration"] = meta.attrib.get("value", "")

            usage = {}
            filaments= {}
            filamentId = 1
            for plate in root.findall(".//plate"):
              for filament in plate.findall(".//filament"):
                used_g = filament.attrib.get("used_g")
                #filamentId = int(filament.attrib.get("id"))
                
                usage[filamentId] = used_g
                filaments[filamentId] = {"id": filamentId,
                                         "tray_info_idx": filament.attrib.get("tray_info_idx"), 
                                         "type":filament.attrib.get("type"), 
                                         "color": filament.attrib.get("color"), 
                                         "used_g": used_g, 
                                         "used_m":filament.attrib.get("used_m")}
                filamentId += 1

            metadata["filaments"] = filaments
            metadata["usage"] = usage
        else:
          logger.info(f"File '{slice_info_path}' not found in the archive.")
          return {}
        filename = time.strftime('%Y%m%d%H%M%S') + "_" + str(uuid.uuid4())[:8]
        metadata["image"] = filename + ".png"
        metadata["model"] = filename + ".3mf"

        with z.open("Metadata/plate_"+metadata["plateID"]+".png") as source_file:
          with open(os.path.join(os.getcwd(), 'static', 'prints', metadata["image"]), 'wb') as target_file:
              target_file.write(source_file.read())
              
        shutil.copyfile(temp_file_name, os.path.join(os.path.join(os.getcwd(), 'static', 'prints'), metadata["model"]))
        # Check for the Metadata/slice_info.config file
        gcode_path = "Metadata/plate_"+metadata["plateID"]+".gcode"
        if gcode_path in z.namelist():
          with z.open(gcode_path) as gcode_file:
            metadata["filamentOrder"] =  get_filament_order(gcode_file)
        return metadata

  except requests.exceptions.RequestException as e:
    logger.error(f"Error downloading file: {e}")
    return {}
  except zipfile.BadZipFile:
    logger.error("The downloaded file is not a valid 3MF archive.")
    return {}
  except ET.ParseError:
    logger.error("Error parsing the XML file.")
    return {}
  except Exception as e:
    logger.error(f"An unexpected error occurred: {e}")
    return {}
