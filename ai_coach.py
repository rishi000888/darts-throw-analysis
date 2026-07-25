"""
Darts Throw Analysis AI — "Ask AI" coaching box.

Two ways to answer a question about a throw's analysis:

- Rule-based ("quick answers"): a small keyword router that explains the
  already-computed numbers in plain language. Free, instant, no API key,
  but can't answer anything outside its keyword list.
- LLM ("AI chat"): a real call to the Claude API, given the computed
  analysis as context. Free-form, but needs ANTHROPIC_API_KEY set and
  costs a small amount per question.

Which mode is used is a per-request choice made by the caller (the
frontend toggle) — this module doesn't decide that itself.
"""

import os

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")


class CoachError(Exception):
    """Raised when a question can't be answered (bad mode, missing key, API error)."""


def _throw_summary_text(throw):
    elbow = throw["right_elbow"]
    wrist = throw["right_wrist"]
    overall = throw["overall"]
    direction = elbow.get("direction") or {}
    lines = [
        f"Throw #{throw['throw_number']}:",
        f"  Right elbow stability: {elbow['score']}% ({elbow['label']}), "
        f"angle ranged {elbow['angle_min']}-{elbow['angle_max']} deg (avg {elbow['angle_avg']}).",
        f"  Elbow movement: {direction.get('summary', 'not enough data')}.",
        f"  Right wrist snap: {wrist['score']}% ({wrist['label']}), "
        f"peak speed {wrist['peak_speed_pct_per_sec']} (% of frame diagonal per second).",
        f"  Release: frame {throw['release_frame']} at {throw['release_time']}s.",
        f"  Overall score: {overall['score']}% ({overall['label']}).",
    ]
    return "\n".join(lines)


def _analysis_context_text(analysis):
    parts = [_throw_summary_text(t) for t in analysis["throws"]]
    comparison = analysis["comparison"]
    parts.append(
        f"\nAcross all {analysis['throw_count']} throw(s): average score "
        f"{comparison['average_score']}% ({comparison['average_label']}), "
        f"best throw #{comparison['best_throw_number']}, "
        f"worst throw #{comparison['worst_throw_number']}, "
        f"consistency {comparison['consistency_score']}% ({comparison['consistency_label']})."
    )
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Rule-based mode
# --------------------------------------------------------------------------

def rule_based_answer(question, analysis):
    q = question.lower()
    throws = analysis["throws"]
    best = max(throws, key=lambda t: t["overall"]["score"])
    worst = min(throws, key=lambda t: t["overall"]["score"])

    # Direction/movement keywords are checked before the generic "elbow" keyword,
    # since almost every direction question also mentions "elbow" (e.g. "which
    # way does my elbow move") and would otherwise never reach this branch.
    if any(w in q for w in ("direction", "left", "right", "up", "down", "move", "drift", "sideways")):
        e = best["right_elbow"]
        direction = e.get("direction")
        if not direction:
            return "Not enough tracked frames to tell which way the elbow moved on this throw."
        return (
            f"On throw #{best['throw_number']}, {direction['summary'].lower()} "
            f"— a drift of about {direction['drift_pct']}% of the frame diagonal from "
            f"the start of the throw to release."
        )

    if any(w in q for w in ("elbow",)):
        e = best["right_elbow"]
        direction = (e.get("direction") or {}).get("summary", "no clear direction detected")
        return (
            f"On your best throw (#{best['throw_number']}), elbow stability scored "
            f"{e['score']}% ({e['label']}). The angle ranged from {e['angle_min']} to "
            f"{e['angle_max']} degrees. {direction}. A tighter angle range usually means "
            f"a more repeatable throw."
        )

    if any(w in q for w in ("wrist", "snap", "release")):
        w_ = best["right_wrist"]
        return (
            f"On your best throw (#{best['throw_number']}), wrist snap scored "
            f"{w_['score']}% ({w_['label']}), with a peak speed of "
            f"{w_['peak_speed_pct_per_sec']} (measured as % of the frame diagonal per second). "
            f"Release happened at frame {best['release_frame']} ({best['release_time']}s)."
        )

    if any(w in q for w in ("consisten", "compare", "throws", "average", "best", "worst")):
        c = analysis["comparison"]
        return (
            f"You threw {analysis['throw_count']} time(s) in this clip. Average overall score: "
            f"{c['average_score']}% ({c['average_label']}). Best was throw #{c['best_throw_number']}, "
            f"worst was throw #{c['worst_throw_number']}. Consistency across throws: "
            f"{c['consistency_score']}% ({c['consistency_label']}) — higher means your scores "
            f"varied less between throws."
        )

    if any(w in q for w in ("score", "overall", "how did i do", "good")):
        o = best["overall"]
        return (
            f"Your best throw (#{best['throw_number']}) scored {o['score']}% overall "
            f"({o['label']}), combining elbow stability and wrist snap in equal parts."
        )

    return (
        "I can answer questions about: elbow stability/movement, wrist snap/release, "
        "which direction your elbow moved, how your throws compare to each other, "
        "or your overall score. Try asking about one of those, or switch to AI Chat "
        "mode for open-ended questions."
    )


# --------------------------------------------------------------------------
# LLM mode
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a darts throwing-technique coach. You'll be given computed pose-analysis "
    "numbers for one or more throws (elbow angle stability, elbow drift direction, wrist "
    "snap speed, release timing, and an overall score), plus a player's question. Answer "
    "the question directly and concisely (2-4 sentences unless more detail is clearly "
    "needed). Only use the numbers you're given — don't invent measurements. These are "
    "heuristic scores from video pose tracking, not a certified biomechanical analysis, "
    "so hedge appropriately if the player asks for absolute certainty."
)


def llm_answer(question, analysis):
    if not ANTHROPIC_SDK_AVAILABLE:
        raise CoachError("The anthropic package isn't installed on the server.")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise CoachError(
            "AI Chat needs an Anthropic API key on the server — set the ANTHROPIC_API_KEY "
            "environment variable and restart the app. Use Quick Answers mode until then."
        )

    client = anthropic.Anthropic()
    context = _analysis_context_text(analysis)

    try:
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Throw analysis:\n{context}\n\nQuestion: {question}",
            }],
        )
    except anthropic.APIStatusError as err:
        raise CoachError(f"AI Chat request failed: {err.message}") from err
    except anthropic.APIConnectionError as err:
        raise CoachError("Couldn't reach the Claude API — check the server's network connection.") from err

    if response.stop_reason == "refusal":
        raise CoachError("The AI declined to answer that question.")

    text = next((block.text for block in response.content if block.type == "text"), "")
    return text or "(no response text)"


def answer_question(question, analysis, mode):
    question = (question or "").strip()
    if not question:
        raise CoachError("Ask a question first.")
    if mode == "llm":
        return llm_answer(question, analysis)
    return rule_based_answer(question, analysis)
