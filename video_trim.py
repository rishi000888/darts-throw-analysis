"""
Darts Throw Analysis AI — trim a video down to a shorter clip.

Pose detection runs on every frame of a video, so a long clip — especially
one with more than one person taking turns throwing — takes far longer to
analyze than the few seconds actually worth scoring, and the auto
throw-detection in pose_analysis.py has no way to know which thrower's
motion to keep. Trimming to just one player's segment before analyzing
fixes both problems.

Uses the static ffmpeg binary bundled by imageio-ffmpeg rather than
requiring a system ffmpeg install. Re-encodes (doesn't stream-copy) so the
cut point lands exactly on the requested time rather than the nearest
keyframe, and always outputs H.264/AAC in an MP4 container so the result
plays back in the browser's <video> element the same as any other clip
already in the library.
"""

import subprocess

import imageio_ffmpeg


class TrimError(Exception):
    """Raised when a clip can't be trimmed — bad range or an ffmpeg failure."""


def trim(src_path, dest_path, start, end):
    if start < 0 or end <= start:
        raise TrimError("End time must be after start time.")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y",
        "-ss", f"{start:.3f}",
        "-i", src_path,
        "-t", f"{end - start:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        dest_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise TrimError(f"ffmpeg failed: {result.stderr.strip()[-500:]}")
