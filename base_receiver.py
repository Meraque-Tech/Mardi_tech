import json
import time
import serial
from serial.tools import list_ports

FIX_STATE = {0: "NO_FIX", 1: "DEAD_RECKONING", 2: "2D", 3: "3D", 4: "GNSS_DR", 5: "TIME_ONLY"}
CORR_AGE = ["N/A", "0-1s", "1-2s", "2-5s", "5-10s", "10-15s", "15-20s",
            "20-30s", "30-45s", "45-60s", "60-90s", "90-120s", ">120s"]

KNOWN_USB_IDS = [(0x10C4, 0xEA60), (0x1A86, 0x7523), (0x0403, 0x6001), (0x303A, 0x1001)]


def rtkState(pvt):
    if pvt["fix"] < 2:
        return "NO_FIX"
    if pvt["carr"] == 2:
        return "RTK_FIXED"
    if pvt["carr"] == 1:
        return "RTK_FLOAT"
    if pvt["diff"]:
        return "DGNSS"
    return "3D_GNSS"


def findEsp32Port():
    for portInfo in list_ports.comports():
        if (portInfo.vid, portInfo.pid) in KNOWN_USB_IDS:
            return portInfo.device
    return None


class RoverGnssReader:
    def __init__(self, port=None, baud=115200, staleAfter=3.0):
        self.fixedPort = port
        self.baud = baud
        self.staleAfter = staleAfter
        self.latest = None
        self.lastUpdate = 0.0
        self.serialConn = None

    def connect(self):
        while self.serialConn is None:
            port = self.fixedPort or findEsp32Port()
            if port is None:
                print("[serial] no ESP32 found, plug it in, retrying")
                time.sleep(2)
                continue
            try:
                self.serialConn = serial.Serial(port, self.baud, timeout=1)
                print(f"[serial] connected on {port}")
            except serial.SerialException as err:
                print(f"[serial] cannot open {port}: {err}, retrying")
                time.sleep(2)

    def disconnect(self):
        if self.serialConn is not None:
            try:
                self.serialConn.close()
            except serial.SerialException:
                pass
            self.serialConn = None

    def isStale(self):
        return (time.monotonic() - self.lastUpdate) > self.staleAfter

    def stream(self):
        self.connect()
        while True:
            try:
                line = self.serialConn.readline().decode("ascii", errors="ignore").strip()
            except (serial.SerialException, OSError):
                print("[serial] connection lost, reconnecting")
                self.disconnect()
                self.connect()
                continue
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "pvt":
                msg["rtkState"] = rtkState(msg)
                msg["fixLabel"] = FIX_STATE.get(msg["fix"], "UNKNOWN")
                msg["corrAgeLabel"] = CORR_AGE[msg["corrAge"]] if msg["corrAge"] < len(CORR_AGE) else "?"
                msg["timestamp"] = time.monotonic()
                self.latest = msg
                self.lastUpdate = msg["timestamp"]
            yield msg


if __name__ == "__main__":
    reader = RoverGnssReader()
    for msg in reader.stream():
        if msg.get("type") == "pvt":
            print(f"{msg['rtkState']:10s} lat={msg['lat']:.9f} lon={msg['lon']:.9f} "
                  f"sats={msg['sats']} hAcc={msg['hacc']:.3f}m corr={msg['corrAgeLabel']}")
        else:
            print(f"[status] {msg.get('msg')}")