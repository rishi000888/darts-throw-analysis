"""
Darts Throw Analysis AI — scheduled local sync for the training archive.

The other half of the two-sided archive design (see library_export.py's
module docstring for the full picture). `app.py`'s analyze routes archive
a session immediately, in-process, but *only* when running under
desktop.py (DARTS_DESKTOP_MODE=1) — that's the only context where "an
analysis just finished" has real filesystem access to this machine. A
browser hitting the deployed Cloud Run service has no such access at all:
the analysis runs on a remote container with no persistent disk and no
route back to Rishi's own PC.

This script is how *those* sessions get archived: it polls the deployed
backend's REST API for anything already analyzed that isn't in the local
archive yet, downloads the video(s), and calls the exact same
`library_export.export_session()` the desktop app uses — so a session gets
an identical PDF/JSON regardless of which path archived it.

Why "scheduled" and not "once a day": confirmed 2026-09-14 that a Cloud
Run redeploy wipes the service's local disk (library.json + every
uploaded video) entirely — see adhor_agent.md's running log. A redeploy
can land at any time Adhor or anyone else pushes to `main`. Running this
once a day risks losing anything uploaded+analyzed in the gap before a
same-day redeploy; running it every couple of hours instead shrinks that
window to something small rather than eliminating it outright (nothing
short of the desktop-app path is a *zero*-risk window).

As of 2026-09-15, this scheduled run is a background safety net, not
Rishi's primary way of syncing — the desktop app now has its own "Sync
Now" button (`/api/library/sync-now` in app.py) that calls this module's
`sync()` on demand, since Rishi's actual play sessions don't happen on a
fixed timer and he'd rather trigger this himself right after a session
than wait up to 2 hours for it.

A Windows Task Scheduler task named "DartsAnalysis Library Sync" was
created 2026-09-14 to run this every 2 hours (`schtasks /query /tn
"DartsAnalysis Library Sync" /v` to inspect it, or open Task Scheduler and
look under the root folder) — battery-power restrictions were explicitly
disabled on that task since this runs on a laptop and the whole point is
not missing a sync window. Its action was switched 2026-09-15 from
python.exe to **pythonw.exe** — python.exe is a console-subsystem binary,
so Task Scheduler launching it with no existing console to attach to was
popping open a new, visible PowerShell/Windows Terminal-looking window on
every run; pythonw.exe is the windowless variant, built for exactly this
kind of background/GUI-less run. One consequence: under pythonw.exe,
sys.stdout/sys.stderr are `None` (confirmed by direct test, no console at
all to attach to) — the win32 stdout/stderr `.reconfigure()` calls above
are guarded for that; `print()` itself is a no-op when sys.stdout is None
rather than an error, so `_log()`'s console output line is silently
skipped on scheduled runs — the `.sync_log.txt` file (see `_log()` below)
is what actually captures scheduled-run output, exactly as it already had
to for the no-console python.exe case. If that task ever needs
recreating:

    schtasks /create /tn "DartsAnalysis Library Sync" /sc hourly /mo 2 ^
        /tr "\"<path to venv>\\Scripts\\pythonw.exe\" \"<path to this file>\"" ^
        /st 00:00
    (then also disable "Start the task only if the computer is on AC
    power" in the task's Conditions tab, or via
    New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries
    -DontStopIfGoingOnBatteries in PowerShell)

Each run appends to `.sync_log.txt` (also inside the archive root) since
Task Scheduler runs this with no console attached — that's the only way
to see what a scheduled run actually did after the fact.

Progress is tracked in a local manifest (`.sync_manifest.json`, inside the
archive root) so re-running this is always safe/idempotent — already
-archived sessions are skipped, and a run that's interrupted partway
through just picks up where it left off next time (the manifest is saved
after each individual session, not once at the end).

Dedup rule for videos that are also part of a two-angle session: a video
referenced as any session's side_video_id/front_video_id is treated as
"belongs to that session" and is never separately archived as its own
single-video entry, even if it also happens to carry its own standalone
`analysis` (e.g. the side clip was analyzed alone before being paired into
a session later that day) — otherwise the same footage would show up
twice in the archive under two different session folders. This is a
judgment call, not something pose_analysis.py enforces; if that ever turns
out to be wrong for how Rishi actually uses the app, change this rule
here, not in the backend.
"""

import argparse
import json
import os
import sys
import tempfile

import requests

import library_export

# Video/session display names are free text (whatever Rishi named the
# recording, including emoji — a real one showed up in local testing) but
# this script's own stdout is often a Windows console stuck on a legacy
# codepage that can't encode them, which would otherwise crash the whole
# sync on a single print() rather than just skipping that one item.
#
# When this runs via the scheduled Task Scheduler task (pythonw.exe, no
# console attached at all — switched from python.exe 2026-09-15 to stop a
# console window popping up on every run), sys.stdout/sys.stderr are None
# rather than a real stream, so .reconfigure() itself would raise
# AttributeError before sync() ever starts — confirmed by direct test.
# print() with sys.stdout is None is fine on its own (Python no-ops it),
# so the only thing that needs guarding is this reconfigure call.
if sys.platform == "win32":
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr is not None:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_BASE_URL = "https://darts-analysis-352937407000.asia-southeast1.run.app"
MANIFEST_NAME = ".sync_manifest.json"
LOG_NAME = ".sync_log.txt"


def _hide_on_windows(path):
    """Best-effort: mark our own bookkeeping files hidden so Rishi's
    Explorer view of the archive root shows only the dated session
    folders he actually cares about, not internal sync state. A leading
    dot means nothing to Windows Explorer the way it does on Unix."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        FILE_ATTRIBUTE_HIDDEN = 0x2
        ctypes.windll.kernel32.SetFileAttributesW(str(path), FILE_ATTRIBUTE_HIDDEN)
    except Exception:
        pass

_log_path = None  # set by sync() — Task Scheduler runs this with no console
                   # attached, so stdout alone would be invisible; a log
                   # file alongside the archive lets Rishi (or a future
                   # agent) actually see what a scheduled run did.


def _log(message):
    from datetime import datetime as _dt
    line = f"[{_dt.now().isoformat(timespec='seconds')}] {message}"
    print(line)
    if _log_path:
        try:
            with open(_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            _hide_on_windows(_log_path)
        except OSError:
            pass  # logging is a nice-to-have, never worth failing a sync over


def _load_manifest(root):
    path = os.path.join(root, MANIFEST_NAME)
    if not os.path.exists(path):
        return {"archived_video_ids": [], "archived_session_ids": []}
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {"archived_video_ids": [], "archived_session_ids": []}
    data.setdefault("archived_video_ids", [])
    data.setdefault("archived_session_ids", [])
    return data


def _save_manifest(root, manifest):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, MANIFEST_NAME)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, path)  # atomic-ish swap so a crash mid-write can't corrupt the manifest
    _hide_on_windows(path)


def _download(base_url, video_url, dest_path, session):
    resp = session.get(base_url.rstrip("/") + video_url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            f.write(chunk)


def sync(base_url=DEFAULT_BASE_URL, root=None, dry_run=False):
    global _log_path
    root = root or library_export.LIBRARY_ROOT
    os.makedirs(root, exist_ok=True)
    _log_path = os.path.join(root, LOG_NAME)
    manifest = _load_manifest(root)
    http = requests.Session()

    _log(f"Syncing from {base_url} into {root} ...")

    sessions = http.get(f"{base_url}/api/sessions", timeout=30).json()
    videos = http.get(f"{base_url}/api/videos", timeout=30).json()
    videos_by_id = {v["id"]: v for v in videos}

    # Any video that's part of ANY session (analyzed or not) is considered
    # "belongs to a session" — see module docstring's dedup rule.
    paired_video_ids = set()
    for s in sessions:
        paired_video_ids.add(s.get("side_video_id"))
        paired_video_ids.add(s.get("front_video_id"))

    archived_now = {"sessions": 0, "videos": 0, "skipped": 0, "errors": 0}

    # --- Two-angle sessions ---
    for s in sessions:
        session_id = s["id"]
        analysis = s.get("dual_analysis")
        if not analysis or not analysis.get("throws"):
            continue
        if session_id in manifest["archived_session_ids"]:
            continue

        side_video = videos_by_id.get(s.get("side_video_id"))
        front_video = videos_by_id.get(s.get("front_video_id"))
        if not side_video or not front_video:
            _log(f"  [skip] session {session_id}: a source video is missing from the library")
            archived_now["skipped"] += 1
            continue

        _log(f"  [session] {session_id} ({s.get('display_name')})")
        if dry_run:
            continue

        with tempfile.TemporaryDirectory(prefix="darts_sync_") as tmp:
            try:
                side_path = os.path.join(tmp, "side" + os.path.splitext(side_video["video_url"])[1])
                front_path = os.path.join(tmp, "front" + os.path.splitext(front_video["video_url"])[1])
                _download(base_url, side_video["video_url"], side_path, http)
                _download(base_url, front_video["video_url"], front_path, http)
                dest = library_export.export_session(
                    {"side": side_path, "front": front_path}, analysis, "two_angle", root=root,
                )
                _log(f"    archived -> {dest}")
                manifest["archived_session_ids"].append(session_id)
                _save_manifest(root, manifest)
                archived_now["sessions"] += 1
            except Exception as exc:
                _log(f"    ERROR archiving session {session_id}: {exc}")
                archived_now["errors"] += 1

    # --- Single-video analyses ---
    for v in videos:
        video_id = v["id"]
        analysis = v.get("analysis")
        if not analysis or not analysis.get("throws"):
            continue
        if video_id in manifest["archived_video_ids"]:
            continue
        if video_id in paired_video_ids:
            continue  # belongs to a session, archived above (or will be once that session is analyzed)

        _log(f"  [video] {video_id} ({v.get('display_name')})")
        if dry_run:
            continue

        with tempfile.TemporaryDirectory(prefix="darts_sync_") as tmp:
            try:
                video_path = os.path.join(tmp, "video" + os.path.splitext(v["video_url"])[1])
                _download(base_url, v["video_url"], video_path, http)
                dest = library_export.export_session(
                    {"video": video_path}, analysis, "single", root=root,
                )
                _log(f"    archived -> {dest}")
                manifest["archived_video_ids"].append(video_id)
                _save_manifest(root, manifest)
                archived_now["videos"] += 1
            except Exception as exc:
                _log(f"    ERROR archiving video {video_id}: {exc}")
                archived_now["errors"] += 1

    _log(
        f"Done. Archived {archived_now['sessions']} session(s) + {archived_now['videos']} "
        f"single video(s), skipped {archived_now['skipped']}, {archived_now['errors']} error(s)."
    )
    return archived_now


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Deployed backend base URL.")
    parser.add_argument("--root", default=None, help="Override the local archive root (for testing).")
    parser.add_argument("--dry-run", action="store_true", help="List what would be archived without doing it.")
    args = parser.parse_args()

    result = sync(base_url=args.base_url, root=args.root, dry_run=args.dry_run)
    sys.exit(1 if result["errors"] else 0)
