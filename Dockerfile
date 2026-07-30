FROM python:3.11-slim

# opencv-contrib-python (pulled in by mediapipe) is a non-headless build and
# needs these at import time, or `import cv2` fails with something like
# "libGL.so.1: cannot open shared object file".
#
# libgomp1 is separate from that — it's the GNU OpenMP runtime MediaPipe's
# own compiled library (which bundles a TensorFlow Lite runtime) needs to
# load at all. Without it, PoseLandmarker.create_from_options() fails deep
# inside ctypes.CDLL()/_dlopen() with no video-related error message, which
# is easy to mistake for a memory or video-size problem — it happens on
# every request, instantly, regardless of what's being analyzed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

COPY . .

# The MediaPipe Tasks API model file isn't part of the pip package and is
# gitignored (see README's local-setup instructions) — fetch the same file
# at build time instead of committing a binary to the repo.
RUN mkdir -p models && \
    curl -L -o models/pose_landmarker_lite.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

RUN mkdir -p static/uploads/videos static/uploads/thumbnails

# Render injects $PORT at runtime; shell-form CMD is required so it expands.
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
