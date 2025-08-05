import os

ip = os.getenv("PRINTER_IP", "192.168.1.100")
code = os.getenv("PRINTER_ACCESS_CODE", "000000")

yaml_content = f"""streams:
  bambu:
    - ffmpeg:rtsps://bblp:{code}@{ip}:322/streaming/channels/0
"""

with open("go2rtc.yaml", "w") as f:
    f.write(yaml_content)