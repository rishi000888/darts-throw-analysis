# Darts Throw Analysis AI — Version 2

A web app for uploading dart-throw videos and reviewing your technique with a
frame-accurate, slow-motion video player, plus AI pose-detection scoring.

## What's included

- Landing page with drag-and-drop upload (1–20 videos, MP4/MOV/AVI/WEBM)
- **Record a throw directly from your camera** — no separate camera app
  needed. Opens a live preview via `getUserMedia`, records with
  `MediaRecorder`, and feeds the clip into the same upload pipeline as a
  picked file once you hit "Use This Recording".
- **Add from YouTube** (`youtube_fetch.py`): paste a YouTube link — a
  coaching clip, a pro's technique, one of your own uploads — and it's
  downloaded straight into the library via `yt-dlp`, same as a picked or
  recorded file. Capped at 10 minutes per clip (`YOUTUBE_MAX_DURATION` env
  var to change it) and restricted to pre-muxed formats so no system
  `ffmpeg` install is required.
- **Trim a clip down before analyzing it** (`video_trim.py`): a toolbar
  under the timeline — "Mark Start" / "Mark End" set an in/out point from
  wherever the player's currently paused, then "Create Trimmed Clip" cuts
  that range into a new library entry (the original is left untouched).
  Meant for cutting a long or shared recording (e.g. two players taking
  turns on one camera) down to just the throws worth scoring — pose
  detection runs on every frame, so trimming first is much faster than
  analyzing the whole thing. Uses a bundled ffmpeg binary
  (`imageio-ffmpeg`), no system install required.
- Video library with thumbnail, duration, file size, rename, and delete
- Custom video player: play / pause / stop, frame-by-frame stepping,
  fullscreen, mute, volume
- Draggable timeline with live time and frame counter
- Playback speed slider from **0.10×** to **2.00×** (step 0.05×) plus quick
  presets, applied instantly via `video.playbackRate`
- Video information panel (filename, resolution, FPS, duration, file size) —
  read with OpenCV at upload time
- **AI Throw Analysis** (`pose_analysis.py`): MediaPipe Pose (Tasks API)
  tracks the shoulder/elbow/wrist of **either arm — a Right/Left toggle sits
  above "Start Analysis"** (right-handed by default) — across every frame,
  **auto-detects each individual throw** in the clip (peaks in wrist speed),
  and scores each one separately: elbow stability, elbow drift direction
  (up/down, left/right), wrist snap, release frame, and an overall score —
  each shown with a color-coded band (Excellent / Very Good / Good / Needs
  Work). A pictograph lets you click between detected throws; a comparison
  line shows average/best/worst/consistency across all of them. Results are
  cached per video *and per arm* in `library.json`, so re-selecting an
  already-analyzed throw is instant, but switching the arm toggle and
  re-running always does a fresh pass. The scoring and throw-detection
  formulas are a documented heuristic (see the docstring in
  `pose_analysis.py`), not a trained biomechanical model — there's no
  labeled dart-throw dataset to calibrate against yet, and fast repeated arm
  motion can be over-counted as extra throws.
- **Ask AI** (`ai_coach.py`): a coaching Q&A box with two modes — **Quick
  Answers** (free, instant, keyword-matched explanations of the computed
  numbers) and **AI Chat** (a real LLM call with the analysis as context,
  for open-ended questions). AI Chat is bring-your-own-key and
  bring-your-own-provider: switching it on opens a modal to pick
  **Anthropic (Claude)**, **OpenAI (GPT)**, or **Google (Gemini)** and paste
  in a key for that provider — this is a shared web app, so each visitor
  pays for their own usage rather than sharing the developer's key. The
  provider choice and key are saved per-provider in that browser's
  `localStorage` and sent with each question; the server never stores them.
  Without a key, that mode returns a clear error and Quick Answers still
  works.
- Fully responsive: 3-column layout on desktop, stacked on tablet/phone,
  touch-friendly controls
- Dark, sports-themed UI (blue / green / white accents)

## Project structure

```
DartsAnalysis/
├── app.py                     # Flask backend — the actual app, shared by web + desktop
├── desktop.py                 # Desktop launcher — wraps app.py in a native window
├── docs/                      # PRD and project-planning docs
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── uploads/
│       ├── videos/            # uploaded video files (created at runtime)
│       ├── thumbnails/        # first-frame thumbnails (created at runtime)
│       └── library.json       # lightweight metadata store
└── README.md
```

## Setup

```bash
cd DartsAnalysis
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Pose analysis needs a MediaPipe Pose Landmarker model file, which isn't
bundled with the `mediapipe` package (Tasks API model files are downloaded
separately). Fetch it once into `models/`:

```bash
mkdir -p models
curl -L -o models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

(On Windows PowerShell: `Invoke-WebRequest -Uri <url above> -OutFile models\pose_landmarker_lite.task`.)

The **AI Chat** mode of the Ask AI box needs no server-side setup — each
visitor pastes their own API key into the in-app modal (Quick Answers mode
needs nothing at all). For local dev only, you can set a fallback key per
provider as an environment variable, used when a request doesn't supply its
own key:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # Claude
$env:OPENAI_API_KEY = "sk-..."          # GPT
$env:GEMINI_API_KEY = "AIza..."         # Gemini
```

Each provider's default model can also be overridden with an env var
(`ANTHROPIC_MODEL`, `OPENAI_MODEL`, `GEMINI_MODEL`) if the built-in default
becomes outdated.

## Run — web app

```bash
python app.py
```

Then open **http://localhost:5050** in your browser. Set the `PORT` env var
to run on a different port if 5050 is already in use.

## Run — desktop app

```bash
python desktop.py
```

Opens the same app in a native window (via `pywebview`) instead of a
browser tab, on its own port (8765) so it can run alongside the web app.
**There is no separate desktop codebase** — `desktop.py` only starts the
Flask server and points a window at it. Every route, template, and static
file is the one in `app.py`/`templates/`/`static/`, so any change you make
there is automatically live in both the web app and the desktop app the
next time either is started. Nothing to keep in sync manually.

## Notes

- Pose detection runs synchronously inside the `/api/analyze/<id>` request —
  for a ~45s clip at 30fps this takes roughly a minute on CPU. There's no
  progress bar beyond the "Analyzing…" status text; a longer clip just takes
  longer.
- Only one arm is tracked per analysis (shoulder/elbow/wrist), picked via
  the Right/Left toggle — there's no whole-body or two-arm scoring.
- Camera recording needs browser permission for the camera and, outside of
  `localhost`, an HTTPS origin — `getUserMedia` is blocked on plain HTTP for
  any other host.
- Add from YouTube runs the download synchronously in the request, same as
  upload — the request just hangs until it's done, so keep clips short.
  Only respect this for content you have the right to download (your own
  uploads, or clips whose owner allows it) — YouTube's Terms of Service
  restrict downloading other creators' videos. Private, age-restricted, or
  region-locked videos will fail to download; cloud hosts (e.g. Render) may
  also have their IPs rate-limited or blocked by YouTube, so this is most
  reliable running locally.
- Possible next steps: PDF report export and multi-video side-by-side
  comparison — the video library already supports up to 20 videos, which was
  built with that in mind.
