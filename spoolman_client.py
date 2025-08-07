import requests
from config import SPOOL_SORTING,get_app_setting
import json

def get_spoolman_url_or_abort():
    url = get_app_setting("SPOOLMAN_API_URL", "")
    if not url:
        abort(500, "L'URL de Spoolman n'est pas définie dans les paramètres.")
    return url

def patchExtraTags(spool_id, old_extras, new_extras):
  for key, value in new_extras.items():
    old_extras[key] = value
  try:
    url = get_spoolman_url_or_abort()
    response = requests.patch(f"{url}/spool/{spool_id}", json={
        "extra": old_extras
    })
    response.raise_for_status()
    return response.json()
  except requests.RequestException as e:
    abort(500, f"Erreur lors de la connexion à Spoolman : {e}")
  
def patchLocation(spool_id, ams_id='', tray_id=''):
  location = ''
  ams_name='AMS_'+str(ams_id)
  if get_app_setting("LOCATION_MAPPING","") != '' :
    d = dict(item.split(":", 1) for item in get_app_setting("LOCATION_MAPPING","").split(";"))
    if ams_name in d:
        if ams_id ==100:
            location = d[ams_name]
        else:
            location = d[ams_name] + ' '+ str(tray_id)
  try:
        url = get_spoolman_url_or_abort()
        response = requests.patch(f"{url}/spool/{spool_id}", json={
            "location": location
        })
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        abort(500, f"Erreur lors de la connexion à Spoolman : {e}")

def getSpoolById(spool_id):
  try:
    url = get_spoolman_url_or_abort()
    response = requests.get(f"{url}/spool/{spool_id}", timeout=5)
    response.raise_for_status()
    return response.json()
  except requests.RequestException as e:
    abort(500, f"Erreur lors de la connexion à Spoolman : {e}")


def fetchSpoolList(archived=False):
  archi='?allow_archived=0'
  if archived:
    archi='?allow_archived=1'
  try:
    url = get_spoolman_url_or_abort()
    if SPOOL_SORTING:
        response = requests.get(f"{url}/spool{archi}&sort={SPOOL_SORTING}")
    else:
        response = requests.get(f"{url}/spool{archi}")
    response.raise_for_status()
    return response.json()
  except requests.RequestException as e:
    abort(500, f"Erreur lors de la connexion à Spoolman : {e}")
  
    
  #print(response.status_code)
  #print(response.text)
  return response.json()

def consumeSpool(spool_id, use_weight):
  #print(f'Consuming {use_weight} from spool {spool_id}')
  try:
    url = get_spoolman_url_or_abort()
    response = requests.put(f"{url}/spool/{spool_id}/use", json={
        "use_weight": use_weight
    })
    response.raise_for_status()
    return response.json()
  except requests.RequestException as e:
    abort(500, f"Erreur lors de la connexion à Spoolman : {e}")
    
def reajust_spool(spool_id, new_weight):
    #print(f"Réajuster spool {spool_id} à {new_weight}g")
  try:
    url = get_spoolman_url_or_abort()
    response = requests.patch(f"{url}/spool/{spool_id}", json={
        "remaining_weight": new_weight
    })
    response.raise_for_status()
    return response.json()
  except requests.RequestException as e:
    abort(500, f"Erreur lors de la connexion à Spoolman : {e}")

def archive_spool(spool_id):
    #print(f"Archiver spool {spool_id}, le déplacer en Archives, vider active_tray et mettre remaining_weight à 0")
  try:
    url = get_spoolman_url_or_abort()
    response = requests.patch(f"{url}/spool/{spool_id}", json={
        "archived": True,
        "location": "Archives",
        "remaining_weight": 0,
        "extra": {
            "active_tray": "\"\""
        }
    })
    response.raise_for_status()
    return response.json()
  except requests.RequestException as e:
    abort(500, f"Erreur lors de la connexion à Spoolman : {e}")

def fetchSettings():
  try:
    url = get_spoolman_url_or_abort()
    response = requests.get(f"{url}/setting/")
    response.raise_for_status()
    data = response.json()

    # Extrahiere die Werte aus den relevanten Feldern
    extra_fields_spool = json.loads(data["extra_fields_spool"]["value"])
    extra_fields_filament = json.loads(data["extra_fields_filament"]["value"])
    base_url = data["base_url"]["value"]
    currency = data["currency"]["value"]
    
    settings = {}
    settings["extra_fields_spool"] = extra_fields_spool 
    settings["extra_fields_filament"] = extra_fields_filament
    settings["base_url"] = base_url.replace('"', '')
    settings["currency"] = currency.replace('"', '')
    
    return settings
  except requests.RequestException as e:
    abort(500, f"Erreur lors de la connexion à Spoolman : {e}")