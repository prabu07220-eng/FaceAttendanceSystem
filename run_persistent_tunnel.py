import subprocess
import time
import re
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLOUDFLARED_PATH = os.path.join(BASE_DIR, "cloudflared.exe")
PYTHON_EXE = sys.executable

# 1. Start Flask app.py if not already running on port 5000
print("[1/2] Launching Flask server on http://127.0.0.1:5000...")
app_proc = subprocess.Popen(
    [PYTHON_EXE, os.path.join(BASE_DIR, "app.py")],
    cwd=BASE_DIR
)

time.sleep(3)

# 2. Launch persistent Cloudflare Tunnel
print("[2/2] Connecting persistent Cloudflare Tunnel...")
tunnel_proc = subprocess.Popen(
    [CLOUDFLARED_PATH, "tunnel", "--url", "http://127.0.0.1:5000"],
    cwd=BASE_DIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

public_url = None
start_time = time.time()

for line in tunnel_proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match and not public_url:
        public_url = match.group(0)
        print("\n" + "=" * 65)
        print(f"🎉 PERSISTENT ONLINE DASHBOARD URL: {public_url}")
        print("=" * 65 + "\n")

# Keep process running indefinitely
tunnel_proc.wait()
