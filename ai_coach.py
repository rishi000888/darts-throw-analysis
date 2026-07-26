"""
Darts Throw Analysis AI — "Ask AI" coaching box.

Two ways to answer a question about a throw's analysis:

- Rule-based ("quick answers"): a small keyword router that explains the
  already-computed numbers in plain language. Free, instant, no API key,
  but can't answer anything outside its keyword list.
- LLM ("AI chat"): a real call to one of several providers (Anthropic,
  OpenAI, or Google), given the computed analysis as context. Free-form,
  but needs an API key and costs a small amount per question. Each caller
  brings their own key and picks their own provider (typed into the
  frontend, passed as `api_key` / `provider`) rather than sharing the
  server's — different visitors of a shared deployment already have keys
  for different providers, so the server has no single "right" key to
  fall back on other than local-dev env vars.

Which mode/provider is used is a per-request choice made by the caller
(the frontend toggle) — this module doesn't decide that itself.
"""

import os

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False

try:
    import openai
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

try:
    from google import genai
    from google.genai import errors as genai_errors
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False

DEFAULT_MODELS = {
    "anthropic": os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8"),
    "openai": os.environ.get("OPENAI_MODEL", "gpt-5.1"),
    "gemini": os.environ.get("GEMINI_MODEL", "gemini-3-pro"),
}

PROVIDER_ENV_KEYS = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


class CoachError(Exception):
    """Raised when a question can't be answered (bad mode, missing key, API error)."""


def _throw_summary_text(throw, arm):
    elbow = throw["elbow"]
    wrist = throw["wrist"]
    overall = throw["overall"]
    direction = elbow.get("direction") or {}
    lines = [
        f"Throw #{throw['throw_number']}:",
        f"  {arm.capitalize()} elbow stability: {elbow['score']}% ({elbow['label']}), "
        f"angle ranged {elbow['angle_min']}-{elbow['angle_max']} deg (avg {elbow['angle_avg']}).",
        f"  Elbow movement: {direction.get('summary', 'not enough data')}.",
        f"  {arm.capitalize()} wrist snap: {wrist['score']}% ({wrist['label']}), "
        f"peak speed {wrist['peak_speed_pct_per_sec']} (% of frame diagonal per second).",
        f"  Release: frame {throw['release_frame']} at {throw['release_time']}s.",
        f"  Overall score: {overall['score']}% ({overall['label']}).",
    ]
    return "\n".join(lines)


def _analysis_context_text(analysis):
    arm = analysis.get("arm", "right")
    parts = [_throw_summary_text(t, arm) for t in analysis["throws"]]
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
    arm = analysis.get("arm", "right")
    throws = analysis["throws"]
    best = max(throws, key=lambda t: t["overall"]["score"])
    worst = min(throws, key=lambda t: t["overall"]["score"])

    # Direction/movement keywords are checked before the generic "elbow" keyword,
    # since almost every direction question also mentions "elbow" (e.g. "which
    # way does my elbow move") and would otherwise never reach this branch.
    if any(w in q for w in ("direction", "left", "right", "up", "down", "move", "drift", "sideways")):
        e = best["elbow"]
        direction = e.get("direction")
        if not direction:
            return "Not enough tracked frames to tell which way the elbow moved on this throw."
        return (
            f"On throw #{best['throw_number']}, {direction['summary'].lower()} "
            f"— a drift of about {direction['drift_pct']}% of the frame diagonal from "
            f"the start of the throw to release."
        )

    if any(w in q for w in ("elbow",)):
        e = best["elbow"]
        direction = (e.get("direction") or {}).get("summary", "no clear direction detected")
        return (
            f"On your best throw (#{best['throw_number']}), {arm} elbow stability scored "
            f"{e['score']}% ({e['label']}). The angle ranged from {e['angle_min']} to "
            f"{e['angle_max']} degrees. {direction}. A tighter angle range usually means "
            f"a more repeatable throw."
        )

    if any(w in q for w in ("wrist", "snap", "release")):
        w_ = best["wrist"]
        return (
            f"On your best throw (#{best['throw_number']}), {arm} wrist snap scored "
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


def _resolve_key(provider, api_key):
    if api_key:
        return api_key
    for env_var in PROVIDER_ENV_KEYS[provider]:
        value = os.environ.get(env_var)
        if value:
            return value
    return None


def _anthropic_answer(question, context, key, model):
    if not ANTHROPIC_SDK_AVAILABLE:
        raise CoachError("The anthropic package isn't installed on the server.")

    client = anthropic.Anthropic(api_key=key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Throw analysis:\n{context}\n\nQuestion: {question}",
            }],
        )
    except anthropic.AuthenticationError as err:
        raise CoachError("That API key was rejected by Anthropic — double-check it and try again.") from err
    except anthropic.APIStatusError as err:
        raise CoachError(f"AI Chat request failed: {err.message}") from err
    except anthropic.APIConnectionError as err:
        raise CoachError("Couldn't reach the Claude API — check your network connection.") from err

    if response.stop_reason == "refusal":
        raise CoachError("The AI declined to answer that question.")

    text = next((block.text for block in response.content if block.type == "text"), "")
    return text or "(no response text)"


def _openai_answer(question, context, key, model):
    if not OPENAI_SDK_AVAILABLE:
        raise CoachError("The openai package isn't installed on the server.")

    client = openai.OpenAI(api_key=key)
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Throw analysis:\n{context}\n\nQuestion: {question}"},
            ],
        )
    except openai.AuthenticationError as err:
        raise CoachError("That API key was rejected by OpenAI — double-check it and try again.") from err
    except openai.APIConnectionError as err:
        raise CoachError("Couldn't reach the OpenAI API — check your network connection.") from err
    except openai.APIStatusError as err:
        raise CoachError(f"AI Chat request failed: {err.message}") from err

    text = response.choices[0].message.content
    return text or "(no response text)"


def _gemini_answer(question, context, key, model):
    if not GEMINI_SDK_AVAILABLE:
        raise CoachError("The google-genai package isn't installed on the server.")

    client = genai.Client(api_key=key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=f"Throw analysis:\n{context}\n\nQuestion: {question}",
            config={"system_instruction": SYSTEM_PROMPT, "max_output_tokens": 1024},
        )
    except genai_errors.ClientError as err:
        if getattr(err, "code", None) in (401, 403):
            raise CoachError("That API key was rejected by Google — double-check it and try again.") from err
        raise CoachError(f"AI Chat request failed: {err}") from err
    except genai_errors.APIError as err:
        raise CoachError(f"AI Chat request failed: {err}") from err

    return response.text or "(no response text)"


PROVIDERS = {
    "anthropic": _anthropic_answer,
    "openai": _openai_answer,
    "gemini": _gemini_answer,
}

PROVIDER_LABELS = {"anthropic": "Anthropic", "openai": "OpenAI", "gemini": "Google"}


def llm_answer(question, analysis, provider="anthropic", api_key=None, model=None):
    if provider not in PROVIDERS:
        raise CoachError(f"Unknown AI provider '{provider}'.")

    key = _resolve_key(provider, api_key)
    if not key:
        raise CoachError(f"This needs an API key for {PROVIDER_LABELS[provider]} — add your own key to continue.")

    context = _analysis_context_text(analysis)
    resolved_model = model or DEFAULT_MODELS[provider]
    return PROVIDERS[provider](question, context, key, resolved_model)


def answer_question(question, analysis, mode, provider="anthropic", api_key=None, model=None):
    question = (question or "").strip()
    if not question:
        raise CoachError("Ask a question first.")
    if mode == "llm":
        return llm_answer(question, analysis, provider=provider, api_key=api_key, model=model)
    return rule_based_answer(question, analysis)


# --------------------------------------------------------------------------
# Full AI analysis — a standalone written coaching take on a throw, using
# the same provider dispatch as Ask AI's "AI Chat" mode, just with a fixed
# prompt instead of a typed-in question. Exists because the heuristic
# elbow/wrist scores and labels (see pose_analysis.py) are a documented
# approximation, not a certified rating — this asks the model to actually
# weigh the underlying numbers itself rather than parrot those labels back.
# --------------------------------------------------------------------------

AI_ANALYSIS_PROMPT = (
    "Give a full, standalone coaching write-up for this session — don't just restate the "
    "numbers and their computed labels, interpret them with your own judgment. The elbow/"
    "wrist scores and Excellent/Very Good/Good/Needs Work labels above are from a simple "
    "heuristic formula, not a certified rating, so weigh the raw numbers (angle range, "
    "wrist speed, release timing) yourself rather than treating those labels as ground "
    "truth. For each throw, assess technique in plain coaching language and call out the "
    "single biggest thing to improve. Then summarize consistency across all throws and "
    "suggest one concrete drill or cue to work on next. Keep it concise and actionable."
)


def ai_analyze(analysis, provider="anthropic", api_key=None, model=None):
    return llm_answer(AI_ANALYSIS_PROMPT, analysis, provider=provider, api_key=api_key, model=model)
