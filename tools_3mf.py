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
  print("Downloading 3MF file from cloud...")
  # Download the file and save it to the temporary file
  response = requests.get(url)
  response.raise_for_status()
  destFile.write(response.content)

def encode_custom_hex(filename):
    return ''.join(f"{ord(c):02x}" if c in "/:" else c for c in filename)
    
def download3mfFromFTP(filename, taskname, destFile):
    CHECK_INTERVAL = 5       # secondes
    TIMEOUT = 180             # secondes
    start_time = time.time()
    found_and_stable = False

    logger.info("Downloading 3MF file from FTP...")
    ftp_host = get_app_setting("PRINTER_IP","")
    ftp_user = "bblp"
    ftp_pass = get_app_setting("PRINTER_ACCESS_CODE","")
    taskname = encode_custom_hex(taskname)
    remote_path = f"/cache/{taskname}.gcode.3mf"
    encoded_path = urllib.parse.quote(remote_path)
    url = f"ftps://{ftp_host}{encoded_path}"
    local_path = destFile.name

    logger.debug(f"Waiting for file to appear and stabilize: {url}")
    time.sleep(5)  # ⏳ Attente initiale minimale

    last_size = -1
    stable_count = 0

    c = pycurl.Curl()
    c.setopt(c.URL, url)
    c.setopt(c.USERPWD, f"{ftp_user}:{ftp_pass}")
    c.setopt(c.NOBODY, True)
    c.setopt(c.SSL_VERIFYPEER, 0)
    c.setopt(c.SSL_VERIFYHOST, 0)
    c.setopt(c.FTP_SSL, c.FTPSSL_ALL)
    c.setopt(c.FTPSSLAUTH, c.FTPAUTH_TLS)
    c.setopt(c.CONNECTTIMEOUT, 10)
    c.setopt(c.TIMEOUT, CHECK_INTERVAL)

    try:
        while time.time() - start_time < TIMEOUT:
            try:
                c.perform()
                current_size = int(c.getinfo(c.CONTENT_LENGTH_DOWNLOAD))
                logger.debug(f"📏 Current file size: {current_size} bytes")

                if current_size == last_size:
                    stable_count += 1
                    logger.debug(f"✅ File size stable {stable_count}/3")
                    if stable_count >= 3:
                        found_and_stable = True
                        break
                else:
                    logger.debug("🔁 File size changed. Resetting stability counter.")
                    stable_count = 1
                    last_size = current_size
            except pycurl.error as e:
                logger.debug(f"📭 File not yet accessible ({e}).")
                stable_count = 0
            time.sleep(CHECK_INTERVAL)
    finally:
        c.close()

    if not found_and_stable:
        logger.error("❌ Timed out: file did not stabilize within 3 minutes.")
        return

    logger.info("📥 File is stable. Starting download...")

    with open(local_path, "wb") as f:
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(c.USERPWD, f"{ftp_user}:{ftp_pass}")
        c.setopt(c.WRITEDATA, f)
        c.setopt(c.SSL_VERIFYPEER, 0)
        c.setopt(c.SSL_VERIFYHOST, 0)
        c.setopt(c.FTP_SSL, c.FTPSSL_ALL)
        c.setopt(c.FTPSSLAUTH, c.FTPAUTH_TLS)
        try:
            c.perform()
            logger.info(f"✅ File successfully downloaded into {local_path} ({last_size} bytes).")
        except pycurl.error as e:
            logger.error(f"❌ Download error: {e}")
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
        download3mfFromFTP(url.replace("ftp://", ""), taskname, temp_file)
        #download3mfFromCloud(url, temp_file)
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
        
        logger.info(metadata)

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
