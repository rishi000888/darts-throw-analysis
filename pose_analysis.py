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
- Throw detection: local peaks in (smoothed) wrist speed that clear a
  minimum height and are spaced at least ~1 second apart. This is a
  heuristic, not a trained action-segmentation model — very fast repeat
  throws or a very shaky camera can confuse it.
- Overall score (per throw): elbow and wrist weighted evenly.
"""

import base64
import math
import os
import statistics

import cv2

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

VISIBILITY_THRESHOLD = 0.5
ELBOW_STABILITY_SCALE = 3.0     # degrees of wobble (post-detrend) -> percentage points lost
MIN_VALID_FRAMES = 5            # below this, we don't trust the result
MIN_PLAUSIBLE_ELBOW_ANGLE = 20  # degrees — a real elbow doesn't fold tighter than this mid-throw;
                                 # anything lower is a mistracked frame even if MediaPipe reported it
                                 # confidently (most often the arm briefly lining up with the camera's
                                 # line of sight at release, which 2D landmark math can't represent)

PEAK_HEIGHT_RATIO = 0.55        # a throw's peak speed must clear this fraction of the video's fastest peak —
                                 # raised from 0.35 because slower incidental motion (walking to the board,
                                 # picking darts back up, resetting stance) was clearing too low a bar and
                                 # getting counted as extra throws
MIN_THROW_GAP_SEC = 1.2         # throws must be at least this far apart
PRE_RELEASE_WINDOW_SEC = 0.8    # how far back a throw's window starts, before its release frame
POST_RELEASE_WINDOW_SEC = 0.3   # how far past release a throw's window extends
DIRECTION_MIN_DRIFT_PCT = 1.5   # % of frame diagonal an elbow must move to count as a real drift, not noise


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


def _extract_landmark_stream(video_path, arm):
    """Single pass over the video: returns (fps, width, height,
    elbow_angles, wrist_positions, shoulder_positions, elbow_positions,
    total_frames)."""
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

            frame_index += 1

    cap.release()
    elbow_angles = _median_filter_by_frame(elbow_angles, window=3)
    return fps, width, height, elbow_angles, wrist_pos, elbow_pos, frame_index


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


def _detect_throw_peaks(speeds_smoothed, fps):
    """Greedy non-max-suppression peak picking: tallest peaks first, skip
    any candidate too close to an already-accepted peak."""
    if not speeds_smoothed:
        return []

    global_max = max(v for _, v in speeds_smoothed)
    if global_max <= 0:
        return []
    threshold = global_max * PEAK_HEIGHT_RATIO
    min_gap_frames = max(1, int(fps * MIN_THROW_GAP_SEC))

    # local maxima: higher than both immediate neighbors (or an edge point)
    candidates = []
    for i in range(len(speeds_smoothed)):
        frame, value = speeds_smoothed[i]
        if value < threshold:
            continue
        prev_v = speeds_smoothed[i - 1][1] if i > 0 else -1
        next_v = speeds_smoothed[i + 1][1] if i < len(speeds_smoothed) - 1 else -1
        if value >= prev_v and value >= next_v:
            candidates.append((frame, value))

    candidates.sort(key=lambda c: c[1], reverse=True)
    accepted = []
    for frame, value in candidates:
        if all(abs(frame - af) >= min_gap_frames for af, _ in accepted):
            accepted.append((frame, value))

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


def analyze_throw(video_path, arm="right"):
    """Run pose detection over the video and return a multi-throw analysis.

    `arm` is "right" or "left" — which arm to track (Version 2 only tracks
    one arm per analysis, picked by the caller since a throw could be
    either-handed).

    Raises AnalysisError if MediaPipe isn't installed, `arm` is invalid, or
    too few frames had a confidently-detected arm to compute anything
    meaningful.
    """
    if arm not in ARM_LANDMARKS:
        raise AnalysisError(f"Invalid arm '{arm}' — must be 'left' or 'right'.")

    fps, width, height, elbow_angles, wrist_pos, elbow_pos, total_frames = \
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
    peaks = _detect_throw_peaks(smoothed, fps)

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

    overall_scores = [t["overall"]["score"] for t in throws]
    best_idx = max(range(len(throws)), key=lambda i: overall_scores[i])
    worst_idx = min(range(len(throws)), key=lambda i: overall_scores[i])
    avg_score = round(statistics.mean(overall_scores))
    consistency_stdev = statistics.pstdev(overall_scores) if len(overall_scores) > 1 else 0.0
    consistency_score = round(_clamp(100 - consistency_stdev * 2))

    return {
        "arm": arm,
        "throw_count": len(throws),
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
    }
