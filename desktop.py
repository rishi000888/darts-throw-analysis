"""
Darts Throw Analysis AI — desktop launcher.

Opens the exact same Flask app (`app.py`) as a native window instead of a
browser tab, using pywebview. There is no separate desktop codebase —
this file only starts the server and points a window at it, so every
route, template, and static asset is shared between the web app
(`python app.py`) and the desktop app (`python desktop.py`). A change to
one is automatically a change to the other.
"""

import os
import socket
import threading
import time

# Must be set before `app` is imported — app.py checks this env var at
# request time (inside the analyze routes) to decide whether it's safe to
# write into Rishi's local Teacher Tools folder. Only this launcher runs
# app.py with real, guaranteed access to this machine's filesystem the
# instant an analysis finishes (the deployed Cloud Run service has none at
# all), so this is the one place that flag gets set. See library_export.py
# and sync_library.py for the two-sided archive design this enables.
os.environ.setdefault("DARTS_DESKTOP_MODE", "1")

import webview

from app import app as flask_app

HOST = "127.0.0.1"
PORT = 8765  # separate from app.py's own default (5000) so both can run at once
ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "icon.ico")


def _run_flask():
    flask_app.run(host=HOST, port=PORT, threaded=True, debug=False, use_reloader=False)


def _wait_for_server(host, port, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


if __name__ == "__main__":
    threading.Thread(target=_run_flask, daemon=True).start()
    if not _wait_for_server(HOST, PORT):
        raise RuntimeError(f"Server did not start on {HOST}:{PORT} in time.")

    webview.create_window(
        "Darts Throw Analysis AI", f"http://{HOST}:{PORT}/",
        width=1360, height=900, min_size=(1000, 700),
    )
    webview.start(icon=ICON_PATH)
