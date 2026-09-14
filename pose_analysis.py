"""
Darts Throw Analysis AI — Version 2 pose analysis.

Runs MediaPipe Pose over an uploaded throw video and turns the raw
landmark stream into per-throw scores. A single video can contain several
separate throws (walk up, throw, walk back, throw again) — this module
finds each individual throw inside the clip by looking for peaks in
wrist speed, scores each one on its own, and returns a comparison across
all detected throws.

Scoring is a documented heuristic, not a trained biomechanical model —
there's no labeled dart-throw dataset to calibrate against, so these
formulas are a reasonable first approximation:

- Elbow stability: how little the elbow angle *wobbles* around the throw's
  own natural swing — not how little it moves overall. A proper throw
  swings the elbow angle open by a lot (bent on the backswing, near-
  straight at release), and that big, smooth extension is the throw
  itself, not instability. This scores the leftover jitter after that
  swing is subtracted out (see `_angle_wobble`): a clean extension stays
  close to a smooth line even though its raw range is wide, while a real
  "chicken wing" throw departs from it with erratic angle changes.
- Elbow direction: net elbow drift (up/down, left/right, as seen on
  screen) from the start of the throw window to the release frame —
  flags things like the elbow dropping or drifting sideways mid-throw.
- Angle noise filtering: MediaPipe occasionally mistracks a single frame
  (motion blur at release, a hand briefly occluding the elbow) — a
  median filter (see `_median_filter_by_frame`) absorbs one-off glitches,
  and readings below `MIN_PLAUSIBLE_ELBOW_ANGLE` are dropped outright as
  mistracked rather than scored, since a real elbow doesn't fold that
  tight mid-throw even when MediaPipe reports the landmark confidently.
- Wrist snap: how sharply the wrist accelerates for that throw relative
  to the video's overall average wrist speed. A real snap is a sudden
  spike, not a steady speed.
- Release frame: the frame where wrist speed peaks within that throw's
  window — the wrist is fastest at the moment the dart leaves the hand.
- Throw counting is NOT automatic — an earlier version guessed how many
  throws were in a clip from peaks in wrist speed, but the guess was
  inconsistent (over- and under-counting depending on camera shake,
  walk-up motion, etc.) and confused users. The caller now supplies
  `throw_count` explicitly (how many throws are actually in the video —
  the UI asks, or the user trims the clip to one throw first via
  video_trim.py). This module only decides *where* each of those throws'
  release moments falls: local peaks in (smoothed) wrist speed, tallest
  first, at least ~1 second apart so two points within the same swing
  can't both get picked (see `_detect_throw_peaks`).
- Overall score (per throw): elbow and wrist weighted evenly.
- Elbow trail overlay: `analyze_throw` can optionally render a copy of the
  video with a line tracking the elbow's tracked position drawn on top —
  opt-in via `overlay_output_path` (it costs real extra time: roughly
  doubles total analysis time in testing, since the second pass over the
  video ends up costing about as much as pose detection itself), since a
  caller may not always want it. The trail is scoped *per throw*: it clears
  and restarts at each throw's own window boundary rather than
  accumulating across the whole clip, so multiple throws' paths don't
  overlap into one continuous line. That means it can't be drawn in the
  same pass as pose extraction — throw windows aren't known until *after*
  the complete wrist-speed signal has been extracted and peaks picked from
  it (see `_detect_throw_peaks`) — so overlay rendering is a second,
  lightweight pass (`_render_elbow_trail_overlay`): raw frame reads + cv2
  drawing using the already-extracted elbow positions, no second pose-
  detection pass.

Dual-angle analysis (`analyze_dual_angle`): a single side-on camera
structurally can't see some things, no matter how good the tracking is —
elbow flare/wrist-snap axis look flattened or ambiguous end-on, and
shoulder/hip rotation isn't visible at all when the torso is edge-on to
the lens. Rishi now optionally records a *second* clip, a front-view
camera near the board facing back at him, for the same throw session.
Rather than fuse the two into one 3D pose (which would need real
calibrated multi-camera triangulation — far more than a heuristic tool
warrants), each video is pose-detected and windowed *independently*,
using the exact same single-angle pipeline as `analyze_throw` for the
side video. The two are then cross-referenced only for the handful of
metrics that genuinely need the second angle: elbow flare direction
(front view's left/right axis, which side view can't show at all — side
view's own left/right is really forward/back extension), wrist snap axis
(does the wrist whip sideways during release — invisible from the side),
and shoulder/hip torso alignment plus stance/foot consistency across
throws (not visible from a side-on camera in any useful way). Every
metric the side view already gets right — elbow stability, release
timing, wrist snap magnitude, extension amount — is left untouched,
carried straight through from the existing side-only scoring.

No literal time-sync (clap, flash, timecode) between the two clips is
needed. Each video finds its own `throw_count` release windows completely
independently (same wrist-speed-peak method as always), and throw N in
one video is paired with throw N in the other purely by chronological
order — since both cameras are recording the *same* sequence of throws in
the *same* session, "1st throw, 2nd throw, ..." already lines them up
without needing the two clips' timestamps to agree at all. This is
deliberately simpler than frame-level sync: nothing here compares an
instant in one video to an instant in the other, so a few seconds of
offset between when each camera started recording is irrelevant. The
only thing that can go wrong is the two videos disagreeing on *how many*
throws they found (a shaky clip, a throw one camera missed) — when that
happens, only the throws both agree on get a combined front-view section;
any extra throws still get their full side-only score, just without the
front-view addition (see `angle_mismatch` in the result).
"""

import base64
import math
import os
import statistics
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Version 2 uses MediaPipe's newer Tasks API (PoseLandmarker) — the older
# `mediapipe.solutions.pose` API isn't shipped in mediapipe builds for
# recent Python versions. The Tasks API needs a model file on disk instead
# of bundling one, downloaded once from Google's model zoo:
# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "models", "pose_landmarker_lite.task")

# MediaPipe Pose landmark indices for each arm — caller picks which one to
# track (see `arm` param on `analyze_throw`), since a throw could be
# right- or left-handed.
ARM_LANDMARKS = {
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16},
    "left": {"shoulder": 11, "elbow": 13, "wrist": 15},
}

# Only read for the front-view video's dual-angle metrics (torso alignment,
# stance/foot consistency) — the side view's own shoulder is already part of
# ARM_LANDMARKS above. Both shoulders/hips (not just the throwing side) are
# needed since alignment is a *relationship between* left and right, not a
# single-point measurement.
TORSO_LANDMARKS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_ankle": 27, "right_ankle": 28,
    "left_foot_index": 31, "right_foot_index": 32,
}

VISIBILITY_THRESHOLD = 0.5
# Hips/ankles are harder for MediaPipe to track cleanly than the arm (more
# often partially out of frame or occluded by clothing/stance) — a stricter
# threshold here would silently drop most frames instead of most outliers.
TORSO_VISIBILITY_THRESHOLD = 0.4
ELBOW_STABILITY_SCALE = 3.0     # degrees of wobble (post-detrend) -> percentage points lost
MIN_VALID_FRAMES = 5            # below this, we don't trust the result
MIN_PLAUSIBLE_ELBOW_ANGLE = 20  # degrees — a real elbow doesn't fold tighter than this mid-throw;
                                 # anything lower is a mistracked frame even if MediaPipe reported it
                                 # confidently (most often the arm briefly lining up with the camera's
                                 # line of sight at release, which 2D landmark math can't represent)

MIN_THROW_GAP_SEC = 1.2         # two accepted release moments must be at least this far apart — stops one
                                 # swing's speed peak from being picked twice when throw_count allows it
PRE_RELEASE_WINDOW_SEC = 0.8    # how far back a throw's window starts, before its release frame
POST_RELEASE_WINDOW_SEC = 0.3   # how far past release a throw's window extends
DIRECTION_MIN_DRIFT_PCT = 1.5   # % of frame diagonal an elbow must move to count as a real drift, not noise
MAX_THROW_COUNT = 30            # sanity cap on the user-supplied throw_count — not a detection limit,
                                 # just guards against a garbage/huge value turning into a very slow request

# Dual-angle (front-view) metrics — see module docstring for why these are
# separate, simple heuristics rather than any real 3D pose fusion.
TORSO_TWIST_SQUARE_DEG = 12      # avg shoulder/hip twist at/below this -> "Square" stance banding
TORSO_TWIST_OPEN_DEG = 25        # avg twist at/below this -> "Slightly Open"; above -> "Very Open"
TORSO_TWIST_DRIFT_DEG = 5        # net twist change (window start -> release) below this counts as "steady"
FOOT_CONSISTENCY_SCALE = 6.0     # % of frame diagonal of foot-centroid spread across throws -> points lost
WRIST_SNAP_SUBWINDOW_SEC = 0.15  # how far either side of release to look for lateral wrist drift

TRAIL_COLOR = (0, 210, 255)         # BGR — orange-yellow, reads clearly against most backgrounds/skin tones
TRAIL_HEAD_COLOR = (0, 140, 255)    # BGR — slightly deeper orange for the current-frame elbow marker
TRAIL_LINE_THICKNESS = 3
TRAIL_POINT_RADIUS = 5
TRAIL_MAX_GAP_SEC = 0.5         # don't draw a line segment across a tracking gap longer than this — a
                                 # brief occlusion shouldn't be shown as the elbow teleporting


class AnalysisError(Exception):
    """Raised when the video doesn't have enough usable pose data."""


def _angle_at(a, b, c):
    """Angle at point b (degrees), formed by rays b->a and b->c."""
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.hypot(*ab)
    mag_cb = math.hypot(*cb)
    if mag_ab == 0 or mag_cb == 0:
        return None
    cos_angle = max(-1.0, min(1.0, dot / (mag_ab * mag_cb)))
    return math.degrees(math.acos(cos_angle))


def _score_band(score):
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Very Good"
    if score >= 60:
        return "Good"
    return "Needs Work"


def _clamp(value, low=0.0, high=100.0):
    return max(low, min(high, value))


def _median_filter_by_frame(values_by_frame, window=3):
    """Median-filter a {frame_index: value} stream, in frame order.

    MediaPipe occasionally mis-tracks a single frame — a hand briefly
    occluding the elbow, motion blur right at release — and produces one
    wildly wrong angle in the middle of otherwise-good frames (e.g. a
    momentary drop to a near-zero angle no real elbow could reach in
    1/30th of a second). A median is robust to exactly that: an isolated
    outlier gets replaced by its neighbors, while a real, gradual swing
    passes through unchanged. A moving *average* would instead smear the
    bad frame's influence across its neighbors too."""
    frames = sorted(values_by_frame)
    values = [values_by_frame[f] for f in frames]
    half = window // 2
    filtered = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        filtered.append(statistics.median(values[lo:hi]))
    return dict(zip(frames, filtered))


def _mux_overlay_video(raw_overlay_path, source_video_path, dest_path):
    """Re-encode the raw elbow-trail overlay (written frame-by-frame with
    OpenCV's VideoWriter, which only reliably supports widely-available
    codecs like mp4v, not browser/mobile-playable H.264) into a proper
    H.264/AAC MP4, and mux the original clip's audio back in — OpenCV's
    frame writer only ever sees silent video frames, so without this the
    overlay video would play back muted. Same bundled ffmpeg binary and
    reasoning as video_trim.py's re-encode. `-map 1:a:0?` makes the audio
    track optional so a source clip with no audio doesn't fail the whole
    mux."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y",
        "-i", raw_overlay_path,
        "-i", source_video_path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        dest_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AnalysisError(f"ffmpeg failed muxing the elbow-trail overlay: {result.stderr.strip()[-300:]}")


def _extract_landmark_stream(video_path, arm, track_torso=False):
    """Single pass over the video (the one expensive part — MediaPipe pose
    detection): returns (fps, width, height, elbow_angles, wrist_positions,
    elbow_positions, total_frames, torso_positions). Doesn't know about
    throws or windows at all — those are only computable *after* this
    returns, from the complete wrist-speed signal (see `analyze_throw`),
    which is also why the elbow-trail overlay can't be drawn inline here
    (see `_render_elbow_trail_overlay`).

    `track_torso`, if True, also pulls both shoulders/hips/ankles/feet
    (`TORSO_LANDMARKS`) out of the same per-frame MediaPipe result — no
    second pose-detection pass, just reading more points off a result
    that's already computed. Only the front-view video in a dual-angle
    analysis needs this (see `analyze_dual_angle`); the side view's own
    single-arm tracking is unaffected and this stays False for every
    existing single-video caller."""
    landmarks = ARM_LANDMARKS[arm]
    if not MEDIAPIPE_AVAILABLE:
        raise AnalysisError(
            "Pose detection isn't available on the server (mediapipe not installed)."
        )
    if not os.path.exists(MODEL_PATH):
        raise AnalysisError(
            "Pose detection model file is missing on the server (models/pose_landmarker_lite.task)."
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise AnalysisError("Could not open video file for analysis.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )

    elbow_angles = {}      # frame_index -> angle_degrees
    wrist_pos = {}         # frame_index -> (x_px, y_px)
    elbow_pos = {}         # frame_index -> (x_px, y_px)
    torso_pos = {}         # frame_index -> {landmark_name: (x_px, y_px)}, only if track_torso

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_index * (1000 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                shoulder, elbow, wrist = lm[landmarks["shoulder"]], lm[landmarks["elbow"]], lm[landmarks["wrist"]]

                if (shoulder.visibility >= VISIBILITY_THRESHOLD and
                        elbow.visibility >= VISIBILITY_THRESHOLD and
                        wrist.visibility >= VISIBILITY_THRESHOLD):
                    shoulder_px = (shoulder.x * width, shoulder.y * height)
                    elbow_px = (elbow.x * width, elbow.y * height)
                    wrist_px = (wrist.x * width, wrist.y * height)

                    angle = _angle_at(shoulder_px, elbow_px, wrist_px)
                    if angle is not None and angle >= MIN_PLAUSIBLE_ELBOW_ANGLE:
                        elbow_angles[frame_index] = angle
                    wrist_pos[frame_index] = wrist_px
                    elbow_pos[frame_index] = elbow_px

                if track_torso:
                    frame_torso = {}
                    for name, idx in TORSO_LANDMARKS.items():
                        point = lm[idx]
                        if point.visibility >= TORSO_VISIBILITY_THRESHOLD:
                            frame_torso[name] = (point.x * width, point.y * height)
                    if frame_torso:
                        torso_pos[frame_index] = frame_torso

            frame_index += 1

    cap.release()
    elbow_angles = _median_filter_by_frame(elbow_angles, window=3)
    return fps, width, height, elbow_angles, wrist_pos, elbow_pos, frame_index, torso_pos


def _render_elbow_trail_overlay(video_path, dest_path, elbow_pos, windows, fps, width, height):
    """Second, MediaPipe-free pass over the video: draws a line tracking
    the elbow's already-extracted position, confined to each individual
    throw's own frame window (`windows` — one (start_frame, end_frame)
    pair per scored throw, chronological, non-overlapping). The trail
    clears and restarts at every throw boundary — entering a new window,
    or leaving one for the gap before the next — instead of accumulating
    across the whole clip, so multiple throws' paths never overlap into
    one continuous line, and frames between throws (walking back to the
    line, resetting stance) show no trail at all.

    Only reads raw frames (`cv2.VideoCapture` + `cv2.imread`-style draw
    calls) — no pose detection here, `elbow_pos` was already extracted by
    `_extract_landmark_stream`. Returns True if the overlay video was
    written successfully, False otherwise (best-effort: a codec/ffmpeg
    problem here shouldn't undo the throw scoring, which has already
    succeeded by the time this runs)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    raw_overlay_path = dest_path + ".raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_overlay_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return False

    max_trail_gap_frames = max(1, int(fps * TRAIL_MAX_GAP_SEC))
    trail_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    last_trail_point = None  # (frame_index, x_px, y_px)
    current_window = None
    window_idx = 0

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # advance past any windows already fully behind this frame
        while window_idx < len(windows) and frame_index > windows[window_idx][1]:
            window_idx += 1
        new_window = None
        if window_idx < len(windows) and windows[window_idx][0] <= frame_index <= windows[window_idx][1]:
            new_window = windows[window_idx]
        if new_window != current_window:
            # crossing a throw boundary either way (into a window, out of one
            # into the gap, or into the next window entirely) — reset the trail
            trail_canvas[:] = 0
            last_trail_point = None
            current_window = new_window

        overlay_frame = frame
        if current_window is not None:
            elbow_here = elbow_pos.get(frame_index)
            if elbow_here is not None:
                x, y = int(elbow_here[0]), int(elbow_here[1])
                if (last_trail_point is not None and
                        frame_index - last_trail_point[0] <= max_trail_gap_frames):
                    cv2.line(trail_canvas, (last_trail_point[1], last_trail_point[2]), (x, y),
                             TRAIL_COLOR, TRAIL_LINE_THICKNESS, lineType=cv2.LINE_AA)
                last_trail_point = (frame_index, x, y)

            trail_mask = trail_canvas.any(axis=2)
            if trail_mask.any():
                overlay_frame = frame.copy()
                overlay_frame[trail_mask] = trail_canvas[trail_mask]
            if elbow_here is not None:
                if overlay_frame is frame:
                    overlay_frame = frame.copy()
                cv2.circle(overlay_frame, (int(elbow_here[0]), int(elbow_here[1])),
                           TRAIL_POINT_RADIUS, TRAIL_HEAD_COLOR, -1, lineType=cv2.LINE_AA)

        writer.write(overlay_frame)
        frame_index += 1

    cap.release()
    writer.release()

    success = False
    try:
        _mux_overlay_video(raw_overlay_path, video_path, dest_path)
        success = True
    except AnalysisError:
        pass  # overlay is a bonus visual — a failed mux shouldn't fail the whole analysis
    finally:
        if os.path.exists(raw_overlay_path):
            os.remove(raw_overlay_path)
    return success


def extract_frame_images(video_path, frame_indices, max_dim=768, jpeg_quality=80):
    """Grab specific frames from the video as base64-encoded JPEG strings.

    Used to give an AI analysis request actual visual context instead of
    just the tracked numbers — the elbow/wrist measurements alone can't
    tell an LLM what someone's stance, grip, or release actually looked
    like. Frames are resized down (a vision model doesn't need full
    resolution, and smaller images mean a faster, cheaper request) and
    re-read directly from disk by seeking to each frame index — this is a
    handful of frame grabs, not a second full pose-detection pass.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    images = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if ok:
            images.append(base64.b64encode(buf).decode("ascii"))

    cap.release()
    return images


def _wrist_speed_series(wrist_pos, fps, diagonal):
    """List of (frame_index, speed_pct_diag_per_sec), sorted by frame."""
    frames = sorted(wrist_pos.keys())
    speeds = []
    for prev, cur in zip(frames, frames[1:]):
        dt = (cur - prev) / fps
        if dt <= 0:
            continue
        x0, y0 = wrist_pos[prev]
        x1, y1 = wrist_pos[cur]
        dist = math.hypot(x1 - x0, y1 - y0)
        speed_pct = (dist / diagonal) * 100 / dt
        speeds.append((cur, speed_pct))
    return speeds


def _smooth(speeds, window=3):
    """Light moving-average smoothing, small enough to preserve throw peaks."""
    values = [s for _, s in speeds]
    smoothed = []
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        smoothed.append(statistics.mean(values[lo:hi]))
    return [(speeds[i][0], smoothed[i]) for i in range(len(speeds))]


def _detect_throw_peaks(speeds_smoothed, fps, throw_count):
    """Pick exactly `throw_count` release moments — the user-supplied
    number of throws in the clip (see module docstring: automatic throw
    *counting* was removed, so this function no longer decides how many
    throws there are, only where each one's release moment falls).

    Greedy non-max-suppression: local maxima in the smoothed wrist-speed
    signal, tallest first, each at least MIN_THROW_GAP_SEC away from an
    already-accepted one so two points within the same swing can't both
    get picked. If the clip doesn't have `throw_count` distinct local
    maxima spaced that far apart (a very short or unusually noisy clip),
    remaining slots are filled from the single highest points left,
    honoring the same spacing rule — so this can return fewer than
    `throw_count` throws if even that's impossible, but never guesses at
    a *different* count than what was asked for."""
    if not speeds_smoothed:
        return []
    min_gap_frames = max(1, int(fps * MIN_THROW_GAP_SEC))

    def is_far_enough(frame, accepted):
        return all(abs(frame - af) >= min_gap_frames for af, _ in accepted)

    def greedy_pick(candidates, accepted):
        accepted_frames = {f for f, _ in accepted}
        for frame, value in sorted(candidates, key=lambda c: c[1], reverse=True):
            if len(accepted) >= throw_count:
                break
            if frame in accepted_frames:
                continue
            if is_far_enough(frame, accepted):
                accepted.append((frame, value))
                accepted_frames.add(frame)
        return accepted

    # local maxima: higher than both immediate neighbors (or an edge point)
    local_maxima = []
    for i in range(len(speeds_smoothed)):
        frame, value = speeds_smoothed[i]
        prev_v = speeds_smoothed[i - 1][1] if i > 0 else -1
        next_v = speeds_smoothed[i + 1][1] if i < len(speeds_smoothed) - 1 else -1
        if value >= prev_v and value >= next_v:
            local_maxima.append((frame, value))

    accepted = greedy_pick(local_maxima, [])
    if len(accepted) < throw_count:
        accepted = greedy_pick(speeds_smoothed, accepted)

    accepted.sort(key=lambda c: c[0])
    return accepted


def _direction_label(dx, dy, diagonal):
    drift_pct = math.hypot(dx, dy) / diagonal * 100
    min_px_component = diagonal * (DIRECTION_MIN_DRIFT_PCT / 100)

    vertical = "Stable"
    if dy <= -min_px_component:
        vertical = "Up"
    elif dy >= min_px_component:
        vertical = "Down"

    horizontal = "Stable"
    if dx <= -min_px_component:
        horizontal = "Left"
    elif dx >= min_px_component:
        horizontal = "Right"

    if vertical == "Stable" and horizontal == "Stable":
        summary = "Stable — elbow held its position"
    else:
        parts = [p for p in (vertical, horizontal) if p != "Stable"]
        summary = "Elbow drifted " + " and ".join(p.lower() for p in parts) + " during the throw"

    return {
        "vertical": vertical,
        "horizontal": horizontal,
        "drift_pct": round(drift_pct, 1),
        "summary": summary,
    }


def _angle_wobble(window_angles_by_frame):
    """How erratically the elbow angle moves, with the throw's own smooth
    cocked-to-extended swing subtracted out first.

    A proper throw swings the elbow angle open by 60-80+ degrees over the
    window (bent on the backswing, near-straight at release) — that's the
    throw itself, not instability, and raw stdev of the angle can't tell
    the two apart: it penalizes a big, clean extension exactly as much as
    real "chicken wing" flailing, since both produce a wide angle range.
    Fitting a straight line through the window's angle-vs-time and scoring
    the leftover residual isolates the wobble around that swing instead of
    the swing's size — a smooth extension fits the line closely (small
    residual) even though its raw range is large, while an erratic throw
    departs from it (large residual) regardless of range."""
    frames = sorted(window_angles_by_frame)
    ys = [window_angles_by_frame[f] for f in frames]
    n = len(ys)
    if n < 3:
        return 0.0

    xs = list(range(n))
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return statistics.pstdev(ys)

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var_x
    intercept = mean_y - slope * mean_x
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    return statistics.pstdev(residuals)


def _score_throw(window_start, window_end, release_frame, fps, diagonal,
                  elbow_angles, wrist_pos, elbow_pos, overall_mean_speed):
    window_angle_frames = {f: a for f, a in elbow_angles.items() if window_start <= f <= window_end}
    window_angles = list(window_angle_frames.values())
    window_wrist = {f: p for f, p in wrist_pos.items() if window_start <= f <= window_end}

    if len(window_angles) < 3 or len(window_wrist) < 3:
        return None

    angle_wobble = _angle_wobble(window_angle_frames)
    elbow_stability_score = round(_clamp(100 - angle_wobble * ELBOW_STABILITY_SCALE))

    window_speeds = _wrist_speed_series(window_wrist, fps, diagonal)
    peak_speed = max((s for _, s in window_speeds), default=0.0)
    snap_ratio = (peak_speed / overall_mean_speed) if overall_mean_speed > 0 else 1.0
    wrist_snap_score = round(_clamp((snap_ratio - 1) * 40))

    overall_score = round(_clamp(0.5 * elbow_stability_score + 0.5 * wrist_snap_score))

    # Elbow direction: start of window vs release frame (nearest tracked frame)
    elbow_frames_in_window = sorted(f for f in elbow_pos if window_start <= f <= window_end)
    direction = None
    if elbow_frames_in_window:
        start_f = elbow_frames_in_window[0]
        end_f = min(elbow_frames_in_window, key=lambda f: abs(f - release_frame))
        x0, y0 = elbow_pos[start_f]
        x1, y1 = elbow_pos[end_f]
        direction = _direction_label(x1 - x0, y1 - y0, diagonal)

    return {
        "window_start_frame": window_start,
        "window_end_frame": window_end,
        "release_frame": release_frame,
        "release_time": round(release_frame / fps, 3),
        "elbow": {
            "angle_avg": round(statistics.mean(window_angles), 1),
            "angle_min": round(min(window_angles), 1),
            "angle_max": round(max(window_angles), 1),
            "score": elbow_stability_score,
            "label": _score_band(elbow_stability_score),
            "direction": direction,
        },
        "wrist": {
            "peak_speed_pct_per_sec": round(peak_speed, 1),
            "score": wrist_snap_score,
            "label": _score_band(wrist_snap_score),
        },
        "overall": {
            "score": overall_score,
            "label": _score_band(overall_score),
        },
    }


def analyze_throw(video_path, arm="right", throw_count=None, overlay_output_path=None):
    """Run pose detection over the video and return a multi-throw analysis.

    `arm` is "right" or "left" — which arm to track (Version 2 only tracks
    one arm per analysis, picked by the caller since a throw could be
    either-handed).

    `throw_count` is required — how many throws are actually in this video.
    Automatic throw counting used to guess this from wrist-speed peaks, but
    the guess was inconsistent (over/under-counting depending on camera
    shake, walk-up motion, throw spacing) and confused users, so it was
    removed; the caller must know or ask (or trim the clip to one throw
    first — see video_trim.py). This function still uses wrist-speed peaks
    to find *where* each of those `throw_count` throws is, just not to
    decide how many there are.

    `overlay_output_path`, if given, is where a copy of the video gets
    rendered with a line tracking the elbow's movement drawn on top,
    separately per throw (see module docstring and
    `_render_elbow_trail_overlay`) — this is opt-in because it costs real
    extra time on top of pose detection itself, so pass it only when the
    caller actually wants the overlay. Best-effort: if rendering fails,
    analysis still succeeds without an overlay video.

    Raises AnalysisError if MediaPipe isn't installed, `arm` or
    `throw_count` is invalid, or too few frames had a confidently-detected
    arm to compute anything meaningful.
    """
    if arm not in ARM_LANDMARKS:
        raise AnalysisError(f"Invalid arm '{arm}' — must be 'left' or 'right'.")

    if throw_count is None:
        raise AnalysisError(
            "How many throws are in this video? Automatic throw counting was "
            "removed because it was unreliable — enter the number of throws "
            "(or trim the clip down to one throw first) and try again."
        )
    try:
        throw_count = int(throw_count)
    except (TypeError, ValueError):
        raise AnalysisError("throw_count must be a whole number.")
    if throw_count < 1:
        raise AnalysisError("throw_count must be at least 1.")
    if throw_count > MAX_THROW_COUNT:
        raise AnalysisError(f"throw_count can't be more than {MAX_THROW_COUNT} for a single video.")

    fps, width, height, elbow_angles, wrist_pos, elbow_pos, total_frames, _torso_pos = \
        _extract_landmark_stream(video_path, arm)
    diagonal = math.hypot(width, height)

    if len(elbow_angles) < MIN_VALID_FRAMES or len(wrist_pos) < MIN_VALID_FRAMES:
        raise AnalysisError(
            f"Couldn't reliably detect a {arm} arm in this video — try a clearer, "
            "well-lit clip with the whole arm in frame."
        )

    all_speeds = _wrist_speed_series(wrist_pos, fps, diagonal)
    if not all_speeds:
        raise AnalysisError("Not enough wrist motion detected to analyze the throw.")

    overall_mean_speed = statistics.mean(s for _, s in all_speeds)
    smoothed = _smooth(all_speeds, window=3)
    peaks = _detect_throw_peaks(smoothed, fps, throw_count)

    if not peaks:
        # Fall back to the single fastest moment as one throw, rather than failing outright.
        peaks = [max(all_speeds, key=lambda s: s[1])]

    pre_frames = int(fps * PRE_RELEASE_WINDOW_SEC)
    post_frames = int(fps * POST_RELEASE_WINDOW_SEC)

    throws = []
    for i, (release_frame, _peak_val) in enumerate(peaks):
        window_start = max(0, release_frame - pre_frames)
        # don't let this throw's window reach back into the previous throw's window
        if i > 0:
            prev_release = peaks[i - 1][0]
            window_start = max(window_start, prev_release + 1)
        window_end = min(total_frames - 1, release_frame + post_frames)

        throw = _score_throw(window_start, window_end, release_frame, fps, diagonal,
                              elbow_angles, wrist_pos, elbow_pos, overall_mean_speed)
        if throw is not None:
            throw["throw_number"] = len(throws) + 1
            throws.append(throw)

    if not throws:
        raise AnalysisError("Detected motion but couldn't score any individual throw reliably.")

    overlay_video_available = False
    if overlay_output_path:
        # Throw windows only exist now, after peaks were picked from the
        # complete wrist-speed signal — see _render_elbow_trail_overlay's
        # docstring for why this has to be a second pass.
        windows = [(t["window_start_frame"], t["window_end_frame"]) for t in throws]
        overlay_video_available = _render_elbow_trail_overlay(
            video_path, overlay_output_path, elbow_pos, windows, fps, width, height
        )

    overall_scores = [t["overall"]["score"] for t in throws]
    best_idx = max(range(len(throws)), key=lambda i: overall_scores[i])
    worst_idx = min(range(len(throws)), key=lambda i: overall_scores[i])
    avg_score = round(statistics.mean(overall_scores))
    consistency_stdev = statistics.pstdev(overall_scores) if len(overall_scores) > 1 else 0.0
    consistency_score = round(_clamp(100 - consistency_stdev * 2))

    return {
        "arm": arm,
        "throw_count": len(throws),
        "throw_count_requested": throw_count,
        "throws": throws,
        "comparison": {
            "average_score": avg_score,
            "average_label": _score_band(avg_score),
            "best_throw_number": throws[best_idx]["throw_number"],
            "worst_throw_number": throws[worst_idx]["throw_number"],
            "consistency_score": consistency_score,
            "consistency_label": _score_band(consistency_score),
        },
        "frames_analyzed": total_frames,
        "frames_with_pose": len(wrist_pos),
        "overlay_video_available": overlay_video_available,
    }


# --------------------------------------------------------------------------
# Dual-angle analysis — cross-referencing a second, front-view video against
# the side view for the handful of metrics a side-on camera can't see (see
# module docstring for the full rationale and why no time-sync is needed).
# --------------------------------------------------------------------------

def _line_angle_deg(p_a, p_b):
    """Angle of the line through two points, in degrees, screen space."""
    return math.degrees(math.atan2(p_b[1] - p_a[1], p_b[0] - p_a[0]))


def _shoulder_hip_twist(frame_torso):
    """How rotated the hips are relative to the shoulders in this frame, in
    degrees — 0 means the shoulder-line and hip-line point the same way on
    screen (a square, face-on stance), larger means the torso is twisted.
    Only meaningful from a front-on camera: viewed from the side, both
    lines project onto nearly the same point and any real twist is
    invisible, which is exactly why this is a front-view-only metric (see
    module docstring)."""
    ls, rs = frame_torso.get("left_shoulder"), frame_torso.get("right_shoulder")
    lh, rh = frame_torso.get("left_hip"), frame_torso.get("right_hip")
    if not (ls and rs and lh and rh):
        return None
    diff = abs(_line_angle_deg(ls, rs) - _line_angle_deg(lh, rh)) % 180
    return min(diff, 180 - diff)


def _torso_label(avg_twist):
    if avg_twist <= TORSO_TWIST_SQUARE_DEG:
        return "Square"
    if avg_twist <= TORSO_TWIST_OPEN_DEG:
        return "Slightly Open"
    return "Very Open"


def _torso_alignment(window_torso, release_frame):
    """Average/range of shoulder-hip twist across a throw's front-view
    window, plus how much it changed from the start of the window to
    release — e.g. a thrower who starts square but rotates open through
    the release shows a positive net_change_deg."""
    twists = {}
    for frame, values in window_torso.items():
        twist = _shoulder_hip_twist(values)
        if twist is not None:
            twists[frame] = twist
    if len(twists) < 3:
        return None

    values = list(twists.values())
    avg_twist = statistics.mean(values)
    start_frame = min(twists)
    release_nearest = min(twists, key=lambda f: abs(f - release_frame))
    net_change = round(twists[release_nearest] - twists[start_frame], 1)
    label = _torso_label(avg_twist)

    if net_change > TORSO_TWIST_DRIFT_DEG:
        drift_note = "opened up further as the throw progressed"
    elif net_change < -TORSO_TWIST_DRIFT_DEG:
        drift_note = "squared up more by release"
    else:
        drift_note = "held a steady rotation through the throw"

    return {
        "twist_avg_deg": round(avg_twist, 1),
        "twist_min_deg": round(min(values), 1),
        "twist_max_deg": round(max(values), 1),
        "net_change_deg": net_change,
        "label": label,
        "summary": (
            f"Shoulders/hips were {label.lower()} on average "
            f"({round(avg_twist, 1)} degrees of twist) — {drift_note}."
        ),
    }


def _wrist_lateral_snap(window_wrist_pos, release_frame, fps, diagonal):
    """How much the wrist drifts left/right (on screen, front view) in the
    short window right around release — a real sideways whip ("casting")
    during the snap, versus a straight release. This is genuinely
    invisible from a side-on camera, where left/right on screen is really
    forward/back extension, not a lateral axis at all (see module
    docstring)."""
    if not window_wrist_pos:
        return None
    sub_frames_window = int(fps * WRIST_SNAP_SUBWINDOW_SEC)
    sub_frames = [f for f in window_wrist_pos if abs(f - release_frame) <= sub_frames_window]
    if len(sub_frames) < 2:
        sub_frames = list(window_wrist_pos.keys())
    xs = [window_wrist_pos[f][0] for f in sub_frames]
    lateral_range_pct = (max(xs) - min(xs)) / diagonal * 100

    if lateral_range_pct <= DIRECTION_MIN_DRIFT_PCT:
        label = "Straight"
        summary = "Wrist snapped straight through release, no visible sideways whip from the front."
    else:
        label = "Lateral Drift"
        summary = (
            f"Wrist drifted sideways during the snap (about {round(lateral_range_pct, 1)}% of frame "
            "width) — a lateral whip the side view alone can't show."
        )

    return {"lateral_range_pct": round(lateral_range_pct, 1), "label": label, "summary": summary}


def _foot_centroid(frame_torso):
    """Rough foot position for one frame — mean of whichever ankle/foot
    landmarks were visible. Used to compare stance placement *between*
    throws in the same front-view video (the camera doesn't move between
    throws in one recording, so raw pixel spread already reflects how
    consistently the feet land in the same spot)."""
    points = [frame_torso[name] for name in
              ("left_ankle", "right_ankle", "left_foot_index", "right_foot_index")
              if name in frame_torso]
    if not points:
        return None
    return (statistics.mean(p[0] for p in points), statistics.mean(p[1] for p in points))


def _stance_consistency(foot_centroids, diagonal):
    """Session-level score: how consistently the feet land in the same
    spot throw-to-throw, front view only — a side-on camera only shows one
    foot's silhouette in front of/behind the other and can't measure this
    at all. Same normalize-by-diagonal, subtract-from-100 pattern as the
    per-video `comparison.consistency_score`."""
    points = [c for c in foot_centroids if c is not None]
    if len(points) < 2:
        return None

    mean_x = statistics.mean(p[0] for p in points)
    mean_y = statistics.mean(p[1] for p in points)
    spreads_pct = [math.hypot(x - mean_x, y - mean_y) / diagonal * 100 for x, y in points]
    avg_spread_pct = statistics.mean(spreads_pct)
    score = round(_clamp(100 - avg_spread_pct * FOOT_CONSISTENCY_SCALE))
    label = _score_band(score)

    return {
        "score": score,
        "label": label,
        "avg_spread_pct": round(avg_spread_pct, 1),
        "summary": (
            f"Foot placement varied by about {round(avg_spread_pct, 1)}% of the frame between "
            f"throws ({label.lower()} stance consistency)."
        ),
    }


def _score_front_view(window_start, window_end, release_frame, fps, diagonal,
                       elbow_pos, wrist_pos, torso_pos):
    """Front-view-only metrics for one throw's window: elbow flare
    direction (reusing `_direction_label`, same function the side view
    uses for its own up/down + forward/back drift — here it's measuring a
    different, genuinely new axis), wrist lateral snap, and torso
    alignment. Returns (front_section_dict, foot_centroid_at_release)."""
    elbow_frames = sorted(f for f in elbow_pos if window_start <= f <= window_end)
    elbow_flare = None
    if elbow_frames:
        start_f = elbow_frames[0]
        end_f = min(elbow_frames, key=lambda f: abs(f - release_frame))
        x0, y0 = elbow_pos[start_f]
        x1, y1 = elbow_pos[end_f]
        elbow_flare = _direction_label(x1 - x0, y1 - y0, diagonal)

    window_wrist = {f: p for f, p in wrist_pos.items() if window_start <= f <= window_end}
    wrist_snap = _wrist_lateral_snap(window_wrist, release_frame, fps, diagonal) if window_wrist else None

    window_torso = {f: torso_pos[f] for f in torso_pos if window_start <= f <= window_end}
    torso_alignment = _torso_alignment(window_torso, release_frame)

    foot_centroid = None
    torso_frames_in_window = sorted(window_torso.keys())
    if torso_frames_in_window:
        release_nearest = min(torso_frames_in_window, key=lambda f: abs(f - release_frame))
        foot_centroid = _foot_centroid(torso_pos[release_nearest])

    front_section = {
        "elbow_flare_direction": elbow_flare,
        "wrist_lateral_snap": wrist_snap,
        "torso_alignment": torso_alignment,
    }
    return front_section, foot_centroid


def _combine_throw_notes(side_throw, front_section):
    """Plain-language cross-reference between the two angles for one throw
    — only for the metrics that genuinely need both (see module
    docstring). Everything else in `side_throw` (elbow stability, wrist
    snap magnitude, release timing, overall score) is left exactly as the
    side-only pipeline computed it and isn't repeated here."""
    notes = {}

    side_direction = (side_throw.get("elbow") or {}).get("direction")
    front_direction = front_section.get("elbow_flare_direction")
    if side_direction and front_direction:
        notes["elbow"] = (
            f"Side view: {side_direction['summary']} "
            f"Front view: {front_direction['summary']}"
        )
    elif front_direction:
        notes["elbow"] = f"Front view: {front_direction['summary']}"

    wrist_snap = front_section.get("wrist_lateral_snap")
    if wrist_snap:
        notes["wrist"] = wrist_snap["summary"]

    torso = front_section.get("torso_alignment")
    if torso:
        notes["torso"] = torso["summary"]

    return notes


def analyze_dual_angle(side_video_path, front_video_path, arm="right", throw_count=None,
                        overlay_output_path=None):
    """Run the full single-angle pipeline on the side video (unchanged,
    exactly as accurate as `analyze_throw` already is), then add a
    front-view video's own independent throw detection, cross-referenced
    only for the metrics a side-on camera can't give (see module
    docstring). Same `arm`/`throw_count` apply to both videos — they're two
    cameras on the same throw session, so the same arm is thrown and the
    same number of throws should appear in both.

    Raises the same `AnalysisError`s as `analyze_throw` for the side video
    (invalid arm/throw_count, too little tracked motion, etc. — validated
    once, since both videos share the same params), plus an analogous
    error if the front video itself doesn't have enough usable pose data.
    """
    side_analysis = analyze_throw(side_video_path, arm=arm, throw_count=throw_count,
                                   overlay_output_path=overlay_output_path)
    # analyze_throw already validated arm/throw_count above — safe to reuse
    # the coerced int from here on rather than re-validating.
    throw_count = int(throw_count)

    fps_f, width_f, height_f, elbow_angles_f, wrist_pos_f, elbow_pos_f, total_frames_f, torso_pos_f = \
        _extract_landmark_stream(front_video_path, arm, track_torso=True)
    diagonal_f = math.hypot(width_f, height_f)

    if len(elbow_angles_f) < MIN_VALID_FRAMES or len(wrist_pos_f) < MIN_VALID_FRAMES:
        raise AnalysisError(
            f"Couldn't reliably detect a {arm} arm in the front-view video — try a clearer, "
            "well-lit clip with your whole body in frame, facing the camera."
        )

    all_speeds_f = _wrist_speed_series(wrist_pos_f, fps_f, diagonal_f)
    if not all_speeds_f:
        raise AnalysisError("Not enough wrist motion detected in the front-view video to analyze the throw.")

    smoothed_f = _smooth(all_speeds_f, window=3)
    peaks_f = _detect_throw_peaks(smoothed_f, fps_f, throw_count)
    if not peaks_f:
        peaks_f = [max(all_speeds_f, key=lambda s: s[1])]

    pre_frames_f = int(fps_f * PRE_RELEASE_WINDOW_SEC)
    post_frames_f = int(fps_f * POST_RELEASE_WINDOW_SEC)

    front_windows = []
    for i, (release_frame, _peak_val) in enumerate(peaks_f):
        window_start = max(0, release_frame - pre_frames_f)
        if i > 0:
            window_start = max(window_start, peaks_f[i - 1][0] + 1)
        window_end = min(total_frames_f - 1, release_frame + post_frames_f)
        front_windows.append((window_start, window_end, release_frame))

    # Pairing by chronological order, not by time — see module docstring
    # for why no literal sync between the two clips is needed.
    side_throws = side_analysis["throws"]
    pair_count = min(len(side_throws), len(front_windows))

    combined_throws = []
    foot_centroids = []
    for i, side_throw in enumerate(side_throws):
        throw = dict(side_throw)
        if i < pair_count:
            fw_start, fw_end, fw_release = front_windows[i]
            front_section, foot_centroid = _score_front_view(
                fw_start, fw_end, fw_release, fps_f, diagonal_f,
                elbow_pos_f, wrist_pos_f, torso_pos_f,
            )
            throw["front_view"] = front_section
            throw["combined"] = _combine_throw_notes(side_throw, front_section)
            foot_centroids.append(foot_centroid)
        else:
            throw["front_view"] = None
            throw["combined"] = None
        combined_throws.append(throw)

    stance_consistency = _stance_consistency(foot_centroids, diagonal_f)

    result = dict(side_analysis)
    result["throws"] = combined_throws
    result["dual_angle"] = True
    result["front_view_throw_count"] = len(front_windows)
    result["angle_mismatch"] = len(side_throws) != len(front_windows)
    result["stance_consistency"] = stance_consistency
    result["frames_analyzed_front"] = total_frames_f
    result["frames_with_pose_front"] = len(wrist_pos_f)
    return result
