"""
Darts Throw Analysis AI — Phase 1 backend.

This Flask app handles:
    - Serving the single-page app (templates/index.html)
    - Uploading throw videos (1-20 videos, mp4/mov/avi/webm)
    - Extracting basic, non-AI metadata (resolution, fps, duration, size)
      with OpenCV so the Video Information Panel can show real numbers
    - Generating a first-frame thumbnail for each video
    - Listing / renaming / deleting uploaded videos
    - Serving video + thumbnail files to the <video> player

No pose detection, no throw analysis, no AI scoring happens here yet.
The `/api/analyze` route is a stub that returns a "coming soon" message —
that's the hook Phase 2 will replace with real OpenCV/MediaPipe logic.
"""

import json
import os
import uuid
from datetime import datetime, timezone

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

import pose_analysis
import ai_coach

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads", "videos")
THUMB_DIR = os.path.join(BASE_DIR, "static", "uploads", "thumbnails")
LIBRARY_FILE = os.path.join(BASE_DIR, "static", "uploads", "library.json")

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "webm"}
VALID_ARMS = {"left", "right"}
MAX_VIDEOS = 20
MAX_CONTENT_LENGTH = 2000 * 1024 * 1024  # 2 GB total request cap (scaled with MAX_VIDEOS)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# --------------------------------------------------------------------------
# Tiny JSON "library" — stores metadata for every uploaded throw video.
# A real database isn't needed for Phase 1, but this keeps things durable
# across server restarts and easy to swap out later.
# --------------------------------------------------------------------------

def _load_library():
    if not os.path.exists(LIBRARY_FILE):
        return []
    with open(LIBRARY_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_library(entries):
    with open(LIBRARY_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _human_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _extract_metadata(filepath):
    """Read basic video properties with OpenCV. This is plain metadata
    reading, not analysis — no frames are inspected for pose/content."""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (frame_count / fps) if fps else 0

    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration": round(duration, 2),
    }


def _generate_thumbnail(filepath, thumb_path):
    cap = cv2.VideoCapture(filepath)
    success, frame = cap.read()
    if success:
        cv2.imwrite(thumb_path, frame)
    cap.release()
    return success


# --------------------------------------------------------------------------
# Page route
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------------------------------
# API: list videos
# --------------------------------------------------------------------------

@app.route("/api/videos", methods=["GET"])
def list_videos():
    return jsonify(_load_library())


# --------------------------------------------------------------------------
# API: upload videos (1-5 at once, or one at a time)
# --------------------------------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def upload_videos():
    library = _load_library()

    if len(library) >= MAX_VIDEOS:
        return jsonify({"error": f"Library already has the maximum of {MAX_VIDEOS} videos."}), 400

    files = request.files.getlist("videos")
    if not files:
        return jsonify({"error": "No files were sent."}), 400

    remaining_slots = MAX_VIDEOS - len(library)
    if len(files) > remaining_slots:
        return jsonify({
            "error": f"Only {remaining_slots} more video(s) can be added (max {MAX_VIDEOS})."
        }), 400

    created = []
    errors = []

    for file in files:
        if file.filename == "":
            continue
        if not _allowed_file(file.filename):
            errors.append(f"{file.filename}: unsupported format (use MP4, MOV, AVI, or WEBM).")
            continue

        original_name = secure_filename(file.filename)
        ext = original_name.rsplit(".", 1)[1].lower()
        video_id = uuid.uuid4().hex[:12]
        stored_name = f"{video_id}.{ext}"
        stored_path = os.path.join(UPLOAD_DIR, stored_name)
        file.save(stored_path)

        metadata = _extract_metadata(stored_path)
        if metadata is None:
            os.remove(stored_path)
            errors.append(f"{file.filename}: could not be read as a video file.")
            continue

        thumb_name = f"{video_id}.jpg"
        thumb_path = os.path.join(THUMB_DIR, thumb_name)
        _generate_thumbnail(stored_path, thumb_path)

        display_name = os.path.splitext(original_name)[0]
        entry = {
            "id": video_id,
            "display_name": display_name,
            "stored_name": stored_name,
            "video_url": f"/static/uploads/videos/{stored_name}",
            "thumbnail_url": f"/static/uploads/thumbnails/{thumb_name}",
            "file_size": os.path.getsize(stored_path),
            "file_size_readable": _human_size(os.path.getsize(stored_path)),
            "width": metadata["width"],
            "height": metadata["height"],
            "fps": metadata["fps"],
            "frame_count": metadata["frame_count"],
            "duration": metadata["duration"],
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        library.append(entry)
        created.append(entry)

        if len(library) >= MAX_VIDEOS:
            break

    _save_library(library)

    status = 201 if created else 400
    return jsonify({"created": created, "errors": errors}), status


# --------------------------------------------------------------------------
# API: rename a video (display name only — stored file is untouched)
# --------------------------------------------------------------------------

@app.route("/api/videos/<video_id>/rename", methods=["PATCH"])
def rename_video(video_id):
    data = request.get_json(silent=True) or {}
    new_name = (data.get("display_name") or "").strip()
    if not new_name:
        return jsonify({"error": "New name cannot be empty."}), 400

    library = _load_library()
    for entry in library:
        if entry["id"] == video_id:
            entry["display_name"] = new_name
            _save_library(library)
            return jsonify(entry)

    return jsonify({"error": "Video not found."}), 404


# --------------------------------------------------------------------------
# API: delete a video
# --------------------------------------------------------------------------

@app.route("/api/videos/<video_id>", methods=["DELETE"])
def delete_video(video_id):
    library = _load_library()
    target = next((e for e in library if e["id"] == video_id), None)
    if target is None:
        return jsonify({"error": "Video not found."}), 404

    video_path = os.path.join(UPLOAD_DIR, target["stored_name"])
    thumb_path = os.path.join(THUMB_DIR, f"{video_id}.jpg")
    for path in (video_path, thumb_path):
        if os.path.exists(path):
            os.remove(path)

    library = [e for e in library if e["id"] != video_id]
    _save_library(library)
    return jsonify({"deleted": video_id})


# --------------------------------------------------------------------------
# API: analysis — Version 2 (MediaPipe pose detection + throw-mechanics
# scoring, implemented in pose_analysis.py). Results are cached onto the
# library entry so re-selecting an already-analyzed video is instant.
# --------------------------------------------------------------------------

@app.route("/api/analyze/<video_id>", methods=["POST"])
def analyze_video(video_id):
    library = _load_library()
    entry = next((e for e in library if e["id"] == video_id), None)
    if entry is None:
        return jsonify({"error": "Video not found."}), 404

    arm = request.args.get("arm", "right")
    if arm not in VALID_ARMS:
        return jsonify({"error": "arm must be 'left' or 'right'."}), 400

    force = request.args.get("force") == "1"
    cached = entry.get("analysis")
    if cached and cached.get("throws") and cached.get("arm") == arm and not force:
        return jsonify({"status": "ok", "cached": True, "analysis": cached})

    video_path = os.path.join(UPLOAD_DIR, entry["stored_name"])
    try:
        analysis = pose_analysis.analyze_throw(video_path, arm=arm)
    except pose_analysis.AnalysisError as err:
        return jsonify({"status": "error", "message": str(err)}), 422

    entry["analysis"] = analysis
    _save_library(library)

    return jsonify({"status": "ok", "cached": False, "analysis": analysis})


# --------------------------------------------------------------------------
# API: Ask AI — coaching Q&A over an already-analyzed video's numbers
# --------------------------------------------------------------------------

@app.route("/api/coach/<video_id>", methods=["POST"])
def coach(video_id):
    library = _load_library()
    entry = next((e for e in library if e["id"] == video_id), None)
    if entry is None:
        return jsonify({"error": "Video not found."}), 404

    analysis = entry.get("analysis")
    if not analysis or not analysis.get("throws"):
        return jsonify({"error": "Analyze this throw before asking about it."}), 400

    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    mode = data.get("mode", "rule")
    api_key = (data.get("api_key") or "").strip() or None
    if mode not in ("rule", "llm"):
        return jsonify({"error": "mode must be 'rule' or 'llm'."}), 400

    try:
        answer = ai_coach.answer_question(question, analysis, mode, api_key=api_key)
    except ai_coach.CoachError as err:
        return jsonify({"error": str(err)}), 400

    return jsonify({"answer": answer, "mode": mode})


if __name__ == "__main__":
    # 5050, not 5000 — this machine's other local app (Teacher Toolkit)
    # hardcodes port 5000, and whichever process starts first wins that
    # port silently, leaving the other's API calls failing with no obvious
    # cause. Still overridable via the PORT env var (e.g. on Render).
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=True, host="0.0.0.0", port=port, threaded=True)
