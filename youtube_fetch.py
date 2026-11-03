"""
Darts Throw Analysis AI — "Add from YouTube" download helper.

Wraps yt-dlp to pull a single YouTube video's file onto disk so it can be
dropped into the same upload pipeline as a picked or recorded file. Kept in
its own module so app.py doesn't need to know about yt-dlp's options API,
the same separation used for ai_coach.py and pose_analysis.py.

Downloads are capped at MAX_DURATION_SECONDS (default 30 minutes, override
with the YOUTUBE_MAX_DURATION env var) — this is a guard against someone
accidentally queuing up a multi-hour video for a synchronous, single-request
download, not a judgment about how long a useful throw clip is. A full
practice-session recording is expected to come in under this and then get
cut down with the Trim feature (video_trim.py) before analysis — pose
detection itself is still bound by gunicorn's 280s request timeout, which in
practice limits a single /api/analyze call to roughly 2-4 minutes of
footage, well under this import cap. (Raised from the original 10-minute
default 2026-10-26 — that was too tight to even get a longer source video
into the library before trimming it.)

Two things about how the download itself is set up are worth knowing, both
changed 2026-11-03 after "Add from YouTube" started failing on every link:

1. The format selector asks for separate video + audio streams and lets
   yt-dlp merge them, pointed at the static ffmpeg binary imageio-ffmpeg
   already ships (the same one pose_analysis.py and video_trim.py use — no
   system ffmpeg install needed). This module used to restrict itself to a
   pre-muxed "best[ext=mp4]/best" stream precisely to avoid needing ffmpeg,
   but YouTube has since stopped serving progressive video+audio formats
   altogether, so that selector now fails with "Requested format is not
   available" on essentially every video. Height is held to <=720 because
   the only consumer is MediaPipe pose detection, which gains nothing from
   1080p+ and would make a long clip's download slow enough to threaten the
   request timeout. A pre-muxed stream is still the last fallback in the
   chain, so the feature degrades rather than dies if ffmpeg is missing.

2. The downloaded file's path is read back from yt-dlp's own
   `requested_downloads[...]["filepath"]` rather than from
   `prepare_filename(info)`. prepare_filename rebuilds the name from the
   info dict's extension, which is not necessarily the extension the file
   ends up with after a merge — and, worse, it happily returns a path for a
   video that was never downloaded at all (see the `rejection` comment in
   fetch() for how that happens).
"""

import os
import re
from urllib.parse import urlparse

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

MAX_DURATION_SECONDS = int(os.environ.get("YOUTUBE_MAX_DURATION", 1800))

MAX_HEIGHT = int(os.environ.get("YOUTUBE_MAX_HEIGHT", 720))

_YOUTUBE_HOSTS = re.compile(r"(^|\.)(youtube\.com|youtube-nocookie\.com|youtu\.be)$")


class FetchError(Exception):
    """Raised when a URL can't be validated or the video can't be downloaded."""


def _is_youtube_url(url):
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return bool(_YOUTUBE_HOSTS.search(host))


def _format_selector():
    """Prefer H.264 video + m4a audio at <=MAX_HEIGHT, merged to MP4.

    H.264/AAC first because that is what every browser and Android player
    in front of this app can play back without re-encoding; VP9/AV1 would
    still analyze fine but can refuse to preview. The last two fallbacks
    drop the codec and then the height constraint so an unusual video still
    imports at whatever it does offer.
    """
    return (
        f"bv*[height<={MAX_HEIGHT}][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
        f"bv*[height<={MAX_HEIGHT}]+ba/"
        f"b[height<={MAX_HEIGHT}]/"
        f"bv*+ba/b"
    )


def _ffmpeg_location():
    """Path to the bundled static ffmpeg, or None if it isn't available.

    Resolved per call rather than at import time so a broken/absent
    imageio-ffmpeg can never stop app.py from importing this module — the
    same best-effort posture pose_analysis.py takes with the same binary.
    yt-dlp accepts a path to the executable itself (not just a directory)
    as long as the filename contains "ffmpeg", which imageio-ffmpeg's
    platform-suffixed name does.
    """
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    return exe if exe and os.path.exists(exe) else None


def _duration_reason(info_dict):
    """The "too long" message for this video, or None if it's acceptable."""
    duration = info_dict.get("duration")
    if duration and duration > MAX_DURATION_SECONDS:
        return (
            f"That video is about {int(duration // 60)} min long — only clips up to "
            f"{MAX_DURATION_SECONDS // 60} minutes are supported."
        )
    return None


def _downloaded_path(ydl, info):
    """Where yt-dlp actually put the file, or None if it never wrote one.

    yt-dlp records the final path (post-merge, post-processing) on each
    entry of `requested_downloads`; that key is simply absent when nothing
    was downloaded. prepare_filename() is kept only as a fallback for an
    older yt-dlp that doesn't populate it, and is checked against the disk
    by the caller.
    """
    if not info:
        return None
    for download in info.get("requested_downloads") or ():
        path = download.get("filepath") or download.get("_filename")
        if path:
            return path
    return ydl.prepare_filename(info)


def fetch(url, dest_dir, video_id):
    """Download `url` into dest_dir as `{video_id}.<ext>`.

    Returns (stored_path, ext, title). Raises FetchError on any problem —
    bad URL, video too long, or the download itself failing.
    """
    if not YTDLP_AVAILABLE:
        raise FetchError("The yt-dlp package isn't installed on the server.")
    if not _is_youtube_url(url):
        raise FetchError("That doesn't look like a YouTube URL.")

    # yt-dlp treats a match_filter rejection as "skip this entry", not as an
    # error: extract_info() returns the info dict and raises nothing, having
    # downloaded no file. Without capturing the reason here, the duration
    # cap surfaced to the user as the generic "the file wasn't found"
    # message below instead of the real "that video is about N min long"
    # explanation — which is exactly how this looked when reported
    # 2026-11-03.
    rejection = {}

    def match_filter(info_dict, *, incomplete=False):
        reason = _duration_reason(info_dict)
        if reason:
            rejection["reason"] = reason
        return reason

    ydl_opts = {
        "format": _format_selector(),
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(dest_dir, f"{video_id}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "match_filter": match_filter,
    }

    ffmpeg_exe = _ffmpeg_location()
    if ffmpeg_exe:
        ydl_opts["ffmpeg_location"] = ffmpeg_exe

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            stored_path = _downloaded_path(ydl, info)
    except yt_dlp.utils.DownloadError as err:
        raise FetchError(str(err).removeprefix("ERROR: ")) from err

    if rejection.get("reason"):
        raise FetchError(rejection["reason"])

    if not stored_path or not os.path.exists(stored_path):
        raise FetchError("Download finished but the file wasn't found.")

    title = (info or {}).get("title") or "YouTube video"
    ext = os.path.splitext(stored_path)[1].lstrip(".").lower()
    return stored_path, ext, title
