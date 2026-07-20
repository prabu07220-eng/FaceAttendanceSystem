import subprocess
import time
import re
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLOUDFLARED_PATH = os.path.join(BASE_DIR, "cloudflared.exe")
PYTHON_EXE = sys.executable

print("==========================================================")
print("🚀 LAUNCHING ONLINE CLOUD DASHBOARD HOSTING SYSTEM...")
print("==========================================================")

# 1. Start Flask app.py in background
print("[1/2] Starting Flask Web Dashboard server...")
app_proc = subprocess.Popen(
    [PYTHON_EXE, os.path.join(BASE_DIR, "app.py")],
    cwd=BASE_DIR,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

time.sleep(2)

# 2. Launch Cloudflare Tunnel
print("[2/2] Generating secure Public HTTPS Cloud URL...")
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
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        public_url = match.group(0)
        break
    if time.time() - start_time > 25:
        break

if public_url:
    print("\n" + "=" * 65)
    print("🎉 YOUR DASHBOARD IS LIVE ONLINE ACROSS THE WORLD!")
    print("==========================================================")
    print(f"  👉 PUBLIC HTTPS URL : {public_url}")
    print(f"  👉 LOGIN USERNAME   : admin")
    print(f"  👉 LOGIN PASSWORD   : admin123")
    print("==========================================================")
    print("Open this URL on your mobile, laptop, or share it anywhere!")
    print("Press Ctrl+C in this terminal to stop online hosting.\n")
    try:
        tunnel_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping online cloud hosting...")
        tunnel_proc.terminate()
        app_proc.terminate()
else:
    print("Failed to extract Cloudflare public URL. Make sure you are connected to the internet.")
    tunnel_proc.terminate()
    app_proc.terminate()
