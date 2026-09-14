"""
Darts Throw Analysis AI — local training-archive export.

Rishi is building a long-term, cross-session archive of every analyzed
throw video (roughly one upload per day, sometimes a single side-view
clip, sometimes a paired side+front "two-angle" session) so a future agent
can read across it and track real improvement trends over time. This
module turns one already-computed analysis (from `pose_analysis.analyze_throw`
or `analyze_dual_angle`) into that archive's unit of storage: one dated
folder containing the source video(s), a human-readable PDF report, and a
machine-readable JSON file describing the exact same session.

Folder/file convention (fixed — the whole point of an archive is that it
stays consistent for years, so don't change this casually):

    <LIBRARY_ROOT>/<YYYY-MM-DD>_session<N>/
        video.mp4                  (single-video sessions)
        side.mp4, front.mp4        (two-angle sessions — matches the
                                     "Two-Angle Stance Session" feature)
        analysis.pdf
        analysis.json

`<N>` starts at 1 and increments per calendar day, so multiple sessions on
the same day (e.g. a morning and evening practice) each get their own
folder rather than colliding.

Why both a PDF and a JSON for the same session: the PDF is for Rishi to
actually read; the JSON is for a future agent (Rishi Darts Teacher) to
parse across 50+ sessions to find trends without having to re-derive
numbers from PDF text, which would be fragile and slow. Both are built
from the exact same `analysis` dict, so they can never disagree.

Called from two places (see adhor_agent.md's running log for why both
exist, not just one):
  - `app.py`'s `/api/analyze/<id>` and `/api/sessions/<id>/analyze` routes,
    immediately after a *fresh* (non-cached) analysis completes, but only
    when `DARTS_DESKTOP_MODE=1` is set in the environment — i.e. only when
    this process is `desktop.py` running locally on Rishi's own machine,
    which is the only place an analysis's completion has in-process access
    to this machine's filesystem. This is the zero-data-loss path.
  - `sync_library.py`, a separate scheduled catch-up script for anything
    analyzed through the plain browser against the deployed Cloud Run
    service, which has no persistent disk and no access to this machine at
    all — see that script's own docstring.
"""

import base64
import io
import json
import os
import shutil
from datetime import datetime, timezone
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import ai_coach
import pose_analysis

LIBRARY_ROOT = os.environ.get(
    "DARTS_LIBRARY_ROOT",
    r"C:\Users\Rishi\Documents\Teacher Tools\Darts Analysis Video",
)


class ExportError(Exception):
    """Raised when a session can't be archived at all (no scored throws to
    save). Best-effort failures in optional parts — narrative generation,
    key-frame extraction — degrade gracefully instead of raising this."""


# --------------------------------------------------------------------------
# Folder naming
# --------------------------------------------------------------------------

def _session_dir_for_today(root, when=None):
    when = when or datetime.now()
    date_str = when.strftime("%Y-%m-%d")
    os.makedirs(root, exist_ok=True)
    n = 1
    while True:
        session_id = f"{date_str}_session{n}"
        candidate = os.path.join(root, session_id)
        if not os.path.exists(candidate):
            return candidate, session_id, date_str
        n += 1


# --------------------------------------------------------------------------
# Narrative — prefer a real AI write-up if a provider key happens to be
# configured locally, otherwise the always-available rule-based one.
# --------------------------------------------------------------------------

def _narrative_for(analysis):
    for provider in ("anthropic", "openai", "gemini"):
        key = ai_coach._resolve_key(provider, None)
        if not key:
            continue
        try:
            return ai_coach.ai_analyze(analysis, provider=provider), f"llm:{provider}"
        except ai_coach.CoachError:
            continue  # missing SDK, rejected key, API error — fall through to the next provider
    return ai_coach.narrative_report(analysis), "rule_based"


# --------------------------------------------------------------------------
# Key frames — a couple of visual reference points so the PDF isn't pure
# text/numbers. Best-effort: the archive is still useful without these.
# --------------------------------------------------------------------------

def _pick_key_frames(video_path, throws):
    if not throws or not video_path or not os.path.exists(video_path):
        return []
    best = max(throws, key=lambda t: t["overall"]["score"])
    release = best["release_frame"]
    follow = min(best.get("window_end_frame", release), release + 5)

    frames = []
    seen = set()
    for idx, label in ((release, "Release"), (follow, "Follow-through")):
        if idx in seen:
            continue
        seen.add(idx)
        images = pose_analysis.extract_frame_images(video_path, [idx], max_dim=900, jpeg_quality=85)
        if images:
            frames.append((f"{label} (throw #{best['throw_number']}, frame {idx})", images[0]))
    return frames


# --------------------------------------------------------------------------
# JSON — the machine-readable side of the archive.
# --------------------------------------------------------------------------

def _build_json_payload(meta, analysis, narrative, narrative_source):
    return {
        "session_id": meta["session_id"],
        "date": meta["date_str"],
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "video_type": meta["video_type"],  # "single" | "two_angle"
        "videos": meta["videos"],  # {"video": "video.mp4"} or {"side": "...", "front": "..."}
        "arm": analysis.get("arm"),
        "throw_count": analysis.get("throw_count"),
        "stance_captured": meta["video_type"] == "two_angle",
        "narrative": narrative,
        "narrative_source": narrative_source,  # "llm:<provider>" | "rule_based"
        # The complete, unmodified analysis dict from pose_analysis.py — the
        # source of truth for every number. Kept as one nested object rather
        # than flattened so this schema never drifts out of sync with
        # whatever fields pose_analysis.py adds or changes over time.
        "analysis": analysis,
    }


# --------------------------------------------------------------------------
# PDF — the human-readable side of the archive.
# --------------------------------------------------------------------------

def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="DartsBody", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="DartsMeta", parent=styles["Normal"], fontSize=9, leading=13,
        textColor=colors.HexColor("#555555"),
    ))
    styles.add(ParagraphStyle(
        name="DartsCaption", parent=styles["Normal"], fontSize=8, leading=11,
        textColor=colors.HexColor("#555555"), alignment=TA_LEFT, spaceBefore=2, spaceAfter=10,
    ))
    return styles


def _direction_cell(direction):
    if not direction:
        return "Not enough data"
    return direction["summary"]


def _throws_table(throws, styles):
    header = ["Throw", "Elbow Stability", "Elbow Direction", "Wrist Snap", "Release", "Overall"]
    data = [header]
    for t in throws:
        elbow, wrist, overall = t["elbow"], t["wrist"], t["overall"]
        data.append([
            str(t["throw_number"]),
            f"{elbow['score']}% ({elbow['label']})",
            Paragraph(escape(_direction_cell(elbow.get("direction"))), styles["DartsCaption"]),
            f"{wrist['score']}% ({wrist['label']})",
            f"frame {t['release_frame']}\n{t['release_time']}s",
            f"{overall['score']}% ({overall['label']})",
        ])
    table = Table(data, repeatRows=1, colWidths=[0.5 * inch, 1.15 * inch, 1.9 * inch, 1.05 * inch, 0.85 * inch, 1.05 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b3d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _stance_section(analysis, styles):
    """Stance/foot position + shoulder-hip alignment + elbow-flare-direction
    — only ever present for two-angle sessions (see analyze_dual_angle in
    pose_analysis.py). Always emits something, so a reader can tell at a
    glance whether a given day has stance data or not, rather than a
    silently blank/missing section."""
    story = [Paragraph("Stance &amp; Front-View (Two-Angle Only)", styles["Heading2"])]

    if not analysis.get("dual_angle"):
        story.append(Paragraph(
            "Not captured this session — single-angle video only. Record a paired "
            "front-view clip via the Two-Angle Stance Session feature to capture "
            "stance/foot position, shoulder-hip alignment, and elbow-flare direction.",
            styles["DartsBody"],
        ))
        return story

    stance = analysis.get("stance_consistency")
    if stance:
        story.append(Paragraph(
            f"Stance consistency: {stance['score']}% ({stance['label']}) — {escape(stance['summary'])}",
            styles["DartsBody"],
        ))
    else:
        story.append(Paragraph(
            "Stance consistency: not enough matched throws between the two videos to score.",
            styles["DartsBody"],
        ))

    for t in analysis.get("throws") or []:
        combined = t.get("combined")
        if not combined:
            continue
        lines = [f"<b>Throw #{t['throw_number']}:</b>"]
        for key in ("elbow", "wrist", "torso"):
            if key in combined:
                lines.append(f"{key.capitalize()} — {escape(combined[key])}")
        story.append(Paragraph("<br/>".join(lines), styles["DartsBody"]))

    return story


def _key_frames_section(frame_images, styles):
    story = [Paragraph("Key Frames", styles["Heading2"])]
    if not frame_images:
        story.append(Paragraph("(no key frames captured for this session)", styles["DartsBody"]))
        return story

    for label, b64 in frame_images:
        try:
            raw = base64.b64decode(b64)
            buf = io.BytesIO(raw)
            from PIL import Image as PILImage
            with PILImage.open(io.BytesIO(raw)) as im:
                w, h = im.size
            target_w = 3.4 * inch
            target_h = target_w * (h / w) if w else target_w
            buf.seek(0)
            story.append(RLImage(buf, width=target_w, height=target_h))
            story.append(Paragraph(escape(label), styles["DartsCaption"]))
        except Exception:
            continue  # a bad frame shouldn't block the rest of the report
    return story


def _build_pdf(dest_path, meta, analysis, narrative, narrative_source, frame_images):
    styles = _styles()
    doc = SimpleDocTemplate(
        dest_path, pagesize=letter,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title=f"Darts Analysis — {meta['session_id']}",
    )

    story = [Paragraph("Darts Throw Analysis — Session Report", styles["Title"])]

    video_type_label = "Two-angle (side + front)" if meta["video_type"] == "two_angle" else "Single video (side only)"
    story.append(Paragraph(
        f"Date: {meta['date_str']} &nbsp;&nbsp;|&nbsp;&nbsp; Session: {meta['session_id']} "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; Arm tracked: {(analysis.get('arm') or 'right').capitalize()} "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; Video type: {video_type_label}",
        styles["DartsMeta"],
    ))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Per-Throw Scores", styles["Heading2"]))
    story.append(_throws_table(analysis.get("throws") or [], styles))
    story.append(Spacer(1, 0.15 * inch))

    comparison = analysis.get("comparison") or {}
    if comparison:
        story.append(Paragraph("Session Comparison", styles["Heading2"]))
        story.append(Paragraph(
            f"Average overall score: {comparison.get('average_score')}% "
            f"({comparison.get('average_label')}). Best throw: #{comparison.get('best_throw_number')}. "
            f"Worst throw: #{comparison.get('worst_throw_number')}. Consistency: "
            f"{comparison.get('consistency_score')}% ({comparison.get('consistency_label')}).",
            styles["DartsBody"],
        ))
        story.append(Spacer(1, 0.1 * inch))

    story.extend(_stance_section(analysis, styles))
    story.append(Spacer(1, 0.1 * inch))
    story.extend(_key_frames_section(frame_images, styles))
    story.append(Spacer(1, 0.1 * inch))

    source_label = {
        "rule_based": "Rule-based summary (no AI provider key was configured for this session)",
    }.get(narrative_source, f"AI-generated ({narrative_source.split(':', 1)[-1].capitalize()})")
    story.append(Paragraph("Coach's Narrative", styles["Heading2"]))
    story.append(Paragraph(source_label, styles["DartsMeta"]))
    story.append(Spacer(1, 0.05 * inch))
    for paragraph in (narrative or "").split("\n\n"):
        if paragraph.strip():
            story.append(Paragraph(escape(paragraph.strip()).replace("\n", "<br/>"), styles["DartsBody"]))

    doc.build(story)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def export_session(video_paths, analysis, video_type, root=None, when=None):
    """Archive one analyzed session into the local training library.

    `video_paths`: dict of absolute source paths already on disk —
        {"video": path} for a single-video session, or
        {"side": path, "front": path} for a two-angle session.
    `analysis`: the raw dict from `pose_analysis.analyze_throw` or
        `analyze_dual_angle` (unmodified — this is the source of truth
        written into analysis.json as-is).
    `video_type`: "single" or "two_angle" — must match `video_paths`'
        keys; passed explicitly rather than inferred so a caller can't
        accidentally mislabel a session by getting the dict keys wrong.

    Returns the created session folder's absolute path. Raises
    `ExportError` only if there's nothing worth archiving (no scored
    throws) — copying the video(s) and writing the PDF/JSON are the
    priority; narrative generation and key-frame extraction are
    best-effort and degrade gracefully on their own failures rather than
    losing the whole archive over a cosmetic extra.
    """
    if not analysis or not analysis.get("throws"):
        raise ExportError("Nothing to archive — analysis has no scored throws.")
    if video_type not in ("single", "two_angle"):
        raise ExportError(f"video_type must be 'single' or 'two_angle', got {video_type!r}.")

    root = root or LIBRARY_ROOT
    dest_dir, session_id, date_str = _session_dir_for_today(root, when)
    os.makedirs(dest_dir, exist_ok=True)

    stored_videos = {}
    for key, src_path in video_paths.items():
        if not src_path or not os.path.exists(src_path):
            continue
        ext = os.path.splitext(src_path)[1] or ".mp4"
        dest_name = f"{key}{ext}"
        shutil.copy2(src_path, os.path.join(dest_dir, dest_name))
        stored_videos[key] = dest_name

    try:
        narrative, narrative_source = _narrative_for(analysis)
    except Exception:
        narrative, narrative_source = (
            "(Narrative generation failed for this session — the numeric analysis "
            "above is still complete and accurate.)",
            "unavailable",
        )

    primary_key = "side" if "side" in video_paths else "video"
    try:
        frame_images = _pick_key_frames(video_paths.get(primary_key), analysis.get("throws") or [])
    except Exception:
        frame_images = []

    meta = {
        "session_id": session_id,
        "date_str": date_str,
        "video_type": video_type,
        "videos": stored_videos,
    }

    with open(os.path.join(dest_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(_build_json_payload(meta, analysis, narrative, narrative_source), f, indent=2)

    _build_pdf(os.path.join(dest_dir, "analysis.pdf"), meta, analysis, narrative, narrative_source, frame_images)

    return dest_dir
