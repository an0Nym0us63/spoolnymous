import requests
from config import SPOOL_SORTING,get_app_setting
import json
from flask import abort
import traceback
import logging
from exceptions import ApplicationError

logger = logging.getLogger(__name__)

def get_spoolman_url():
    url = get_app_setting("SPOOLMAN_BASE_URL", "")
    if not url or url=="":
        raise ApplicationError("L'URL de Spoolman n'est pas définie dans les paramètres.")
    return f"{url}/api/v1"

def patchExtraTags(spool_id, old_extras, new_extras):
    for key, value in new_extras.items():
        old_extras[key] = value
    try:
        url = get_spoolman_url()
        response = requests.patch(f"{url}/spool/{spool_id}", json={"extra": old_extras},timeout=5)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        logger.error(traceback.format_exc())
        raise ApplicationError(f"Erreur lors de la connexion à Spoolman : {e}")

def patchLocation(spool_id, ams_id='', tray_id=''):
    location = ''
    ams_name = 'AMS_' + str(ams_id)
    try:
        mapping = get_app_setting("LOCATION_MAPPING", "")
        if mapping:
            d = dict(item.split(":", 1) for item in mapping.split(";"))
            if ams_name in d:
                location = d[ams_name] if ams_id == 100 else d[ams_name] + ' ' + str(tray_id)

        url = get_spoolman_url()
        response = requests.patch(f"{url}/spool/{spool_id}", json={"location": location},timeout=5)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        logger.error(traceback.format_exc())
        raise ApplicationError(f"Erreur lors de la connexion à Spoolman : {e}")

def getSpoolById(spool_id):
    try:
        url = get_spoolman_url()
        response = requests.get(f"{url}/spool/{spool_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(traceback.format_exc())
        raise ApplicationError(f"Erreur lors de la connexion à Spoolman : {e}")

def fetchSpoolList(archived=False):
    archi = '?allow_archived=1' if archived else '?allow_archived=0'
    try:
        url = get_spoolman_url()
        sort_param = f"&sort={SPOOL_SORTING}" if SPOOL_SORTING else ""
        response = requests.get(f"{url}/spool{archi}{sort_param}",timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(traceback.format_exc())
        raise ApplicationError(f"Erreur lors de la connexion à Spoolman : {e}")

def consumeSpool(spool_id, use_weight):
    try:
        url = get_spoolman_url()
        response = requests.put(f"{url}/spool/{spool_id}/use", json={"use_weight": use_weight},timeout=5)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        logger.error(traceback.format_exc())
        raise ApplicationError(f"Erreur lors de la connexion à Spoolman : {e}")

def reajust_spool(spool_id, new_weight):
    try:
        url = get_spoolman_url()
        response = requests.patch(f"{url}/spool/{spool_id}", json={"remaining_weight": new_weight},timeout=5)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        logger.error(traceback.format_exc())
        raise ApplicationError(f"Erreur lors de la connexion à Spoolman : {e}")

def archive_spool(spool_id):
    try:
        url = get_spoolman_url()
        response = requests.patch(f"{url}/spool/{spool_id}", json={
            "archived": True,
            "location": "Archives",
            "remaining_weight": 0,
            "extra": {"active_tray": "\"\""}
        },timeout=5)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        logger.error(traceback.format_exc())
        raise ApplicationError(f"Erreur lors de la connexion à Spoolman : {e}")

def fetchSettings():
    try:
        url = get_spoolman_url()
        response = requests.get(f"{url}/setting/",timeout=5)
        response.raise_for_status()
        data = response.json()

        return {
            "extra_fields_spool": json.loads(data["extra_fields_spool"]["value"]),
            "extra_fields_filament": json.loads(data["extra_fields_filament"]["value"]),
            "base_url": data["base_url"]["value"].replace('"', ''),
            "currency": data["currency"]["value"].replace('"', '')
        }
    except requests.RequestException as e:
        logger.error(traceback.format_exc())
        raise ApplicationError(f"Erreur lors de la connexion à Spoolman : {e}")