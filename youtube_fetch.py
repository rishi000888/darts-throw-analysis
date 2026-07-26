"""
Darts Throw Analysis AI — "Add from YouTube" download helper.

Wraps yt-dlp to pull a single YouTube video's file onto disk so it can be
dropped into the same upload pipeline as a picked or recorded file. Kept in
its own module so app.py doesn't need to know about yt-dlp's options API,
the same separation used for ai_coach.py and pose_analysis.py.

Downloads are capped at MAX_DURATION_SECONDS (default 10 minutes, override
with the YOUTUBE_MAX_DURATION env var) — throw clips are short, and this
keeps someone from accidentally queuing up a multi-hour video for a
synchronous, single-request download. The format is restricted to a
progressive (pre-muxed) MP4/best stream rather than yt-dlp's usual
video+audio merge, since that merge needs a standalone ffmpeg binary that
isn't a dependency of this app.
"""

import os
import re
from urllib.parse import urlparse

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

MAX_DURATION_SECONDS = int(os.environ.get("YOUTUBE_MAX_DURATION", 600))

_YOUTUBE_HOSTS = re.compile(r"(^|\.)(youtube\.com|youtube-nocookie\.com|youtu\.be)$")


class FetchError(Exception):
    """Raised when a URL can't be validated or the video can't be downloaded."""


def _is_youtube_url(url):
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return bool(_YOUTUBE_HOSTS.search(host))


def _duration_filter(info_dict, *, incomplete=False):
    duration = info_dict.get("duration")
    if duration and duration > MAX_DURATION_SECONDS:
        return (
            f"That video is about {int(duration // 60)} min long — only clips up to "
            f"{MAX_DURATION_SECONDS // 60} minutes are supported."
        )
    return None


def fetch(url, dest_dir, video_id):
    """Download `url` into dest_dir as `{video_id}.<ext>`.

    Returns (stored_path, ext, title). Raises FetchError on any problem —
    bad URL, video too long, or the download itself failing.
    """
    if not YTDLP_AVAILABLE:
        raise FetchError("The yt-dlp package isn't installed on the server.")
    if not _is_youtube_url(url):
        raise FetchError("That doesn't look like a YouTube URL.")

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": os.path.join(dest_dir, f"{video_id}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "match_filter": _duration_filter,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            stored_path = ydl.prepare_filename(info)
    except yt_dlp.utils.DownloadError as err:
        raise FetchError(str(err).removeprefix("ERROR: ")) from err

    if not os.path.exists(stored_path):
        raise FetchError("Download finished but the file wasn't found.")

    title = info.get("title") or "YouTube video"
    ext = os.path.splitext(stored_path)[1].lstrip(".").lower()
    return stored_path, ext, title
