"""
MQTT Manager — repris de v1 et réécrit proprement.
Gère la connexion à l'imprimante Bambu Lab via MQTT TLS.
Publie l'état dans PrinterState (singleton en mémoire).
"""
import asyncio
import json
import logging
import ssl
import threading
from typing import Callable
import paho.mqtt.client as mqtt
from ..models.printer import PrinterState, AMS, AMSTray

logger = logging.getLogger(__name__)

# Singleton état imprimante
state = PrinterState()
_listeners: list[Callable] = []


def get_state() -> PrinterState:
    return state


def subscribe_state(fn: Callable):
    """Enregistre un callback appelé à chaque mise à jour d'état."""
    _listeners.append(fn)


def _notify():
    for fn in _listeners:
        try:
            fn(state)
        except Exception:
            logger.exception("Listener error")


class MQTTManager:
    def __init__(self):
        self._client: mqtt.Client | None = None
        self._connected = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._printer_id = ""
        self._ip = ""
        self._code = ""

    async def start(self):
        """Démarre en arrière-plan (non-bloquant)."""
        from ..services.settings_service import get_setting
        from ..db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            self._printer_id = await get_setting(db, "PRINTER_ID")
            self._ip = await get_setting(db, "PRINTER_IP")
            self._code = await get_setting(db, "PRINTER_ACCESS_CODE")

        if not self._ip or not self._printer_id:
            logger.info("MQTT: imprimante non configurée, connexion différée")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"MQTT: démarrage vers {self._ip}")

    async def stop(self):
        self._stop_event.set()
        if self._client:
            self._client.disconnect()

    def reconnect(self, ip: str, printer_id: str, code: str):
        """Appelé depuis les settings pour reconnecter avec de nouveaux paramètres."""
        self._ip = ip
        self._printer_id = printer_id
        self._code = code
        if self._client:
            try:
                self._client.disconnect()
            except Exception:
                pass
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self._connect_once()
            except Exception as e:
                logger.warning(f"MQTT connexion échouée: {e}, retry dans 15s")
            self._stop_event.wait(15)

    def _connect_once(self):
        client = mqtt.Client()
        client.username_pw_set("bblp", self._code)

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        client.tls_set_context(ctx)
        client.tls_insecure_set(True)

        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message

        client.connect(self._ip, 8883, keepalive=60)
        self._client = client
        client.loop_forever()

    def _on_connect(self, client, userdata, flags, rc):
        self._connected = True
        state.connected = True
        _notify()
        logger.info(f"MQTT connecté (rc={rc})")
        client.subscribe(f"device/{self._printer_id}/report")
        # Demander l'état complet
        client.publish(
            f"device/{self._printer_id}/request",
            json.dumps({"pushing": {"sequence_id": "0", "command": "pushall"}}),
        )

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        state.connected = False
        _notify()
        logger.info(f"MQTT déconnecté (rc={rc})")

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            self._process(data)
        except Exception:
            logger.exception("MQTT message error")

    def _process(self, data: dict):
        p = data.get("print", {})
        if not p:
            return

        changed = False

        # Statut / progression
        if "gcode_state" in p:
            state.status = p["gcode_state"]
            changed = True
        if "mc_percent" in p:
            state.progress = float(p["mc_percent"])
            changed = True
        if "layer_num" in p:
            state.layer = int(p["layer_num"])
            changed = True
        if "total_layer_num" in p:
            state.total_layers = int(p["total_layer_num"])
            changed = True
        if "mc_remaining_time" in p:
            state.remaining_minutes = int(p["mc_remaining_time"])
            changed = True
        if "subtask_name" in p:
            state.print_name = p["subtask_name"]
            changed = True
        if "job_id" in p:
            state.job_id = str(p["job_id"])
            changed = True

        # Températures (firmwares récents + legacy)
        if "bed_temper" in p:
            state.bed_temp = float(p["bed_temper"])
            changed = True
        if "bed_target_temper" in p:
            state.target_bed_temp = float(p["bed_target_temper"])
            changed = True
        if "nozzle_temper" in p:
            state.nozzle_temp = float(p["nozzle_temper"])
            changed = True
        if "nozzle_target_temper" in p:
            state.target_nozzle_temp = float(p["nozzle_target_temper"])
            changed = True
        if "chamber_temper" in p:
            state.chamber_temp = float(p["chamber_temper"])
            changed = True

        # AMS
        ams_data = p.get("ams", {})
        if "ams" in ams_data:
            state.ams_list = []
            for ams_raw in ams_data["ams"]:
                ams = AMS(id=int(ams_raw.get("id", 0)))
                ams.humidity = int(ams_raw.get("humidity_raw", 0))
                ams.temp = float(ams_raw.get("temp", 0))
                for tray_raw in ams_raw.get("tray", []):
                    tray = AMSTray(id=int(tray_raw.get("id", 0)))
                    tray.color = tray_raw.get("tray_color", "")
                    tray.filament_type = tray_raw.get("tray_sub_brands", "") or tray_raw.get("tray_type", "")
                    tray.remain = int(tray_raw.get("remain", 0))
                    tray.uuid = tray_raw.get("tray_uuid", "")
                    tray.info_idx = tray_raw.get("tray_info_idx", "")
                    tray.empty = not bool(tray_raw.get("tray_sub_brands"))
                    ams.trays.append(tray)
                state.ams_list.append(ams)
            changed = True

        if changed:
            _notify()

    def publish(self, payload: dict) -> bool:
        if not self._client or not self._connected:
            return False
        try:
            self._client.publish(
                f"device/{self._printer_id}/request",
                json.dumps(payload),
            )
            return True
        except Exception:
            return False


mqtt_manager = MQTTManager()
