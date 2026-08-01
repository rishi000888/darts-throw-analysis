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
from datetime import datetime, timedelta, timezone

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory
from google.cloud import storage as gcs
from werkzeug.utils import secure_filename

import pose_analysis
import ai_coach
import youtube_fetch
import video_trim

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

# Cloud Run enforces a hard ~32MB limit on request bodies, which the direct
# /api/upload route above can't get around no matter how big MAX_CONTENT_LENGTH
# is. Large recordings instead go straight to this bucket via a signed URL
# (see /api/upload-url + /api/upload-complete), bypassing Cloud Run's ingress
# entirely for the big file transfer.
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "darts-throw-analysis-videos")
_storage_client = None


def _get_storage_client():
    global _storage_client
    if _storage_client is None:
        _storage_client = gcs.Client()
    return _storage_client

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
# The desktop app (desktop.py) runs this same app with debug=False, which
# would otherwise cache the compiled template in memory for the process's
# whole lifetime — editing templates/index.html wouldn't show up until the
# window was closed and reopened. Force template reload regardless of debug
# mode so a template edit is only ever one page-refresh away.
app.config["TEMPLATES_AUTO_RELOAD"] = True


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


def _build_entry(video_id, stored_path, display_name):
    """Turn a video file already sitting at `stored_path` into a library
    entry — metadata + thumbnail. Shared by the upload and YouTube routes,
    which differ only in how the file got onto disk. Returns None (and
    removes the file) if it can't be read as a video."""
    metadata = _extract_metadata(stored_path)
    if metadata is None:
        os.remove(stored_path)
        return None

    thumb_name = f"{video_id}.jpg"
    thumb_path = os.path.join(THUMB_DIR, thumb_name)
    _generate_thumbnail(stored_path, thumb_path)

    stored_name = os.path.basename(stored_path)
    return {
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


# --------------------------------------------------------------------------
# Page route
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


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

        display_name = os.path.splitext(original_name)[0]
        entry = _build_entry(video_id, stored_path, display_name)
        if entry is None:
            errors.append(f"{file.filename}: could not be read as a video file.")
            continue

        library.append(entry)
        created.append(entry)

        if len(library) >= MAX_VIDEOS:
            break

    _save_library(library)

    status = 201 if created else 400
    return jsonify({"created": created, "errors": errors}), status


# --------------------------------------------------------------------------
# API: large-video upload path (bypasses Cloud Run's ~32MB request limit)
#
# The client asks here first for a short-lived signed URL, PUTs the raw
# video bytes straight to Cloud Storage (never touching this Flask app),
# then calls /api/upload-complete so the server pulls it down locally and
# runs it through the same _build_entry() pipeline as a normal upload.
# --------------------------------------------------------------------------

@app.route("/api/upload-url", methods=["POST"])
def create_upload_url():
    library = _load_library()
    if len(library) >= MAX_VIDEOS:
        return jsonify({"error": f"Library already has the maximum of {MAX_VIDEOS} videos."}), 400

    data = request.get_json(silent=True) or {}
    original_name = secure_filename(data.get("filename") or "video.mp4")
    if not _allowed_file(original_name):
        return jsonify({"error": "Unsupported format (use MP4, MOV, AVI, or WEBM)."}), 400

    ext = original_name.rsplit(".", 1)[1].lower()
    video_id = uuid.uuid4().hex[:12]
    blob_name = f"pending-uploads/{video_id}.{ext}"
    content_type = data.get("content_type") or "application/octet-stream"

    bucket = _get_storage_client().bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    upload_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=20),
        method="PUT",
        content_type=content_type,
    )

    return jsonify({
        "video_id": video_id,
        "blob_name": blob_name,
        "upload_url": upload_url,
        "content_type": content_type,
    }), 200


@app.route("/api/upload-complete/<video_id>", methods=["POST"])
def complete_upload(video_id):
    library = _load_library()
    if len(library) >= MAX_VIDEOS:
        return jsonify({"error": f"Library already has the maximum of {MAX_VIDEOS} videos."}), 400

    data = request.get_json(silent=True) or {}
    blob_name = data.get("blob_name")
    original_name = secure_filename(data.get("filename") or "video.mp4")
    if not blob_name or video_id not in blob_name:
        return jsonify({"error": "Missing or invalid blob_name."}), 400

    bucket = _get_storage_client().bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return jsonify({"error": "Upload not found in storage — it may have expired, please try again."}), 404

    ext = blob_name.rsplit(".", 1)[1].lower()
    stored_name = f"{video_id}.{ext}"
    stored_path = os.path.join(UPLOAD_DIR, stored_name)
    blob.download_to_filename(stored_path)
    blob.delete()

    display_name = os.path.splitext(original_name)[0]
    entry = _build_entry(video_id, stored_path, display_name)
    if entry is None:
        return jsonify({"error": "Uploaded file could not be read as a video."}), 400

    library.append(entry)
    _save_library(library)

    return jsonify({"created": [entry], "errors": []}), 201


# --------------------------------------------------------------------------
# API: add a video by downloading it from YouTube (yt-dlp, youtube_fetch.py)
# --------------------------------------------------------------------------

@app.route("/api/youtube", methods=["POST"])
def add_from_youtube():
    library = _load_library()
    if len(library) >= MAX_VIDEOS:
        return jsonify({"error": f"Library already has the maximum of {MAX_VIDEOS} videos."}), 400

    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Paste a YouTube URL first."}), 400

    video_id = uuid.uuid4().hex[:12]
    try:
        stored_path, _ext, title = youtube_fetch.fetch(url, UPLOAD_DIR, video_id)
    except youtube_fetch.FetchError as err:
        return jsonify({"error": str(err)}), 400

    entry = _build_entry(video_id, stored_path, title)
    if entry is None:
        return jsonify({"error": "That video downloaded but couldn't be read as a video file."}), 400

    library.append(entry)
    _save_library(library)
    return jsonify({"created": [entry]}), 201


# --------------------------------------------------------------------------
# API: trim a video down to a shorter clip (video_trim.py) — adds the
# result as a new library entry rather than replacing the original, so a
# clip with more than one player still has its full recording kept around.
# --------------------------------------------------------------------------

@app.route("/api/videos/<video_id>/trim", methods=["POST"])
def trim_video(video_id):
    library = _load_library()
    entry = next((e for e in library if e["id"] == video_id), None)
    if entry is None:
        return jsonify({"error": "Video not found."}), 404

    if len(library) >= MAX_VIDEOS:
        return jsonify({"error": f"Library already has the maximum of {MAX_VIDEOS} videos."}), 400

    data = request.get_json(silent=True) or {}
    try:
        start = float(data.get("start"))
        end = float(data.get("end"))
    except (TypeError, ValueError):
        return jsonify({"error": "start and end must be numbers, in seconds."}), 400

    if start < 0:
        return jsonify({"error": "Start time can't be negative."}), 400
    if end <= start:
        return jsonify({"error": "End time must be after start time."}), 400

    duration = entry.get("duration") or 0
    if duration and end > duration + 0.5:
        return jsonify({"error": f"End time is past the end of the video ({duration:.2f}s)."}), 400

    src_path = os.path.join(UPLOAD_DIR, entry["stored_name"])
    new_id = uuid.uuid4().hex[:12]
    dest_path = os.path.join(UPLOAD_DIR, f"{new_id}.mp4")

    try:
        video_trim.trim(src_path, dest_path, start, end)
    except video_trim.TrimError as err:
        return jsonify({"error": str(err)}), 400

    new_entry = _build_entry(new_id, dest_path, f"{entry['display_name']} (trimmed)")
    if new_entry is None:
        return jsonify({"error": "Trimmed clip could not be read as a video."}), 400

    library.append(new_entry)
    _save_library(library)
    return jsonify({"created": [new_entry]}), 201


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
    provider = data.get("provider", "anthropic")
    api_key = (data.get("api_key") or "").strip() or None
    model = (data.get("model") or "").strip() or None
    if mode not in ("rule", "llm"):
        return jsonify({"error": "mode must be 'rule' or 'llm'."}), 400
    if mode == "llm" and provider not in ai_coach.PROVIDERS:
        return jsonify({"error": f"Unknown AI provider '{provider}'."}), 400

    try:
        answer = ai_coach.answer_question(question, analysis, mode, provider=provider, api_key=api_key, model=model)
    except ai_coach.CoachError as err:
        return jsonify({"error": str(err)}), 400

    return jsonify({"answer": answer, "mode": mode})


# --------------------------------------------------------------------------
# API: Analyze by AI — a full written coaching take on an already-analyzed
# throw, generated by an LLM instead of the fixed heuristic scoring bands.
# Cached per (video, provider) since it costs a real API call — re-opening
# the same video is instant unless the provider changes or force=1.
# --------------------------------------------------------------------------

MAX_AI_REFERENCE_FRAMES = 5


def _reference_frame_indices(analysis, max_frames=MAX_AI_REFERENCE_FRAMES):
    """Pick a handful of frame indices spanning the best throw's window, to
    attach as images to an AI analysis request — the numbers alone can't
    show an LLM what the throw actually looked like."""
    throws = analysis.get("throws") or []
    if not throws:
        return []

    best_number = analysis.get("comparison", {}).get("best_throw_number")
    throw = next((t for t in throws if t["throw_number"] == best_number), throws[0])
    start, end = throw["window_start_frame"], throw["window_end_frame"]
    if end <= start:
        return [start]

    count = min(max_frames, end - start + 1)
    if count <= 1:
        return [start]
    step = (end - start) / (count - 1)
    return sorted({round(start + i * step) for i in range(count)})


@app.route("/api/ai-analyze/<video_id>", methods=["POST"])
def ai_analyze_video(video_id):
    library = _load_library()
    entry = next((e for e in library if e["id"] == video_id), None)
    if entry is None:
        return jsonify({"error": "Video not found."}), 404

    analysis = entry.get("analysis")
    if not analysis or not analysis.get("throws"):
        return jsonify({"error": "Analyze this throw before requesting an AI analysis."}), 400

    data = request.get_json(silent=True) or {}
    provider = data.get("provider", "anthropic")
    api_key = (data.get("api_key") or "").strip() or None
    model = (data.get("model") or "").strip() or None
    force = request.args.get("force") == "1"

    if provider not in ai_coach.PROVIDERS:
        return jsonify({"error": f"Unknown AI provider '{provider}'."}), 400

    cached = entry.get("ai_analysis")
    if cached and cached.get("provider") == provider and not force:
        return jsonify({"result": cached["text"], "cached": True})

    video_path = os.path.join(UPLOAD_DIR, entry["stored_name"])
    images = pose_analysis.extract_frame_images(video_path, _reference_frame_indices(analysis))

    try:
        text = ai_coach.ai_analyze(analysis, provider=provider, api_key=api_key, model=model, images=images)
    except ai_coach.CoachError as err:
        return jsonify({"error": str(err)}), 400

    entry["ai_analysis"] = {
        "provider": provider,
        "text": text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_library(library)

    return jsonify({"result": text, "cached": False})


if __name__ == "__main__":
    # 5050, not 5000 — this machine's other local app (Teacher Toolkit)
    # hardcodes port 5000, and whichever process starts first wins that
    # port silently, leaving the other's API calls failing with no obvious
    # cause. Still overridable via the PORT env var (e.g. on Render).
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=True, host="0.0.0.0", port=port, threaded=True)
