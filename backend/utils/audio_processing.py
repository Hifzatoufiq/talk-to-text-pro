import os
import re
import json
import logging
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------- OpenAI API ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_PUBLIC") or ""
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")
client = OpenAI(api_key=OPENAI_API_KEY)

# Uploads folder path (shared with app.py)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def extract_audio_from_file(filepath: str) -> str:
    """
    Convert any input audio/video file into mono 16k WAV.
    Tries pydub first, falls back to moviepy if needed.
    """
    filename = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(UPLOAD_FOLDER, filename + "_converted.wav")
    log.info("Extracting audio from: %s -> %s", filepath, out_path)

    # Try pydub first (ffmpeg required)
    try:
        audio = AudioSegment.from_file(filepath)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(out_path, format="wav")
        return out_path
    except Exception as e:
        log.warning("pydub failed, trying moviepy: %s", e)

    # Fallback to moviepy
    try:
        clip = VideoFileClip(filepath)
        if clip.audio is None:
            clip.close()
            raise ValueError("No audio stream in media")
        clip.audio.write_audiofile(out_path, fps=16000, codec="pcm_s16le")
        clip.close()
        return out_path
    except Exception as e:
        log.error("Failed to extract audio: %s", e)
        raise e


def transcribe_audio(filepath: str, language: str = "auto") -> str:
    """
    Transcribe audio with OpenAI Whisper.
    If language == "auto", Whisper will auto-detect the language.
    Otherwise we pass a language hint (e.g., "en", "ur", "hi").
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    try:
        kwargs = {"model": "whisper-1"}
        if language and language.lower() != "auto":
            kwargs["language"] = language

        with open(filepath, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(file=audio_file, **kwargs)

        text = getattr(transcript, "text", None) or str(transcript)
        log.info("Transcription OK (preview): %s", (text[:120] + "...") if len(text) > 120 else text)
        return text
    except Exception as e:
        log.exception("Whisper transcription failed: %s", e)
        return ""


def _code_to_lang_name(code: str) -> str:
    """
    Convert short codes to clean language names for the prompt.
    Auto maps to English (per your rule).
    """
    m = {
        "en": "English",
        "ur": "Urdu",
        "hi": "Hindi",
        "auto": "English",  # IMPORTANT: default to English when Auto is selected
        "english": "English",
        "urdu": "Urdu",
        "hindi": "Hindi",
    }
    return m.get((code or "").lower(), "English")


def _loose_json_extract(text: str) -> str:
    """
    Extract the largest-looking JSON object from a text response.
    This helps when the model wraps JSON with extra prose.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _coerce_structured_notes(ai_text: str, transcript: str) -> dict:
    """
    Try to parse model output into a strict structure.
    Falls back to basic heuristics if JSON parsing fails.
    """
    # 1) Try strict JSON parse
    raw = _loose_json_extract(ai_text)
    try:
        data = json.loads(raw)
        # Normalize/validate keys
        exec_sum = str(data.get("executive_summary", "") or "")
        key_points = list(data.get("key_points", []) or [])
        action_items = list(data.get("action_items", []) or [])
        decisions = list(data.get("decisions", []) or [])
        sentiment = str(data.get("sentiment", "") or "Unknown")

        # Coerce action items to objects with task/owner/due
        coerced_ai = []
        for itm in action_items:
            if isinstance(itm, dict):
                coerced_ai.append({
                    "task": str(itm.get("task", "") or ""),
                    "owner": str(itm.get("owner", "") or ""),
                    "due": str(itm.get("due", "") or ""),
                })
            else:
                # If it's a plain string, put it into task
                coerced_ai.append({"task": str(itm), "owner": "", "due": ""})

        return {
            "executive_summary": exec_sum,
            "key_points": [str(x) for x in key_points][:10],
            "action_items": coerced_ai[:10],
            "decisions": [str(x) for x in decisions][:10],
            "sentiment": sentiment if sentiment in ["Positive", "Neutral", "Negative"] else "Unknown",
        }
    except Exception:
        pass

    # 2) Heuristic fallback
    exec_sum = ai_text.strip()
    # Extract bullets
    import re as _re
    bullets = _re.findall(r"(?m)^[\-\*\u2022]\s+(.*)$", ai_text)
    key_points = [b.strip() for b in bullets][:10]

    # Simple action items
    action_items = []
    for line in ai_text.splitlines():
        if _re.search(r"(?i)\b(action|todo|task)\b", line):
            owner = ""
            due = ""
            owner_m = _re.search(r"(?i)owner\s*[:\-]\s*([^|,;]+)", line)
            due_m = _re.search(r"(?i)due\s*[:\-]\s*([^|,;]+)", line)
            if owner_m: owner = owner_m.group(1).strip()
            if due_m: due = due_m.group(1).strip()
            task = _re.sub(r"(?i)\b(owner|due)\s*[:\-]\s*[^|,;]+", "", line)
            task = _re.sub(r"(?i)\b(action|todo|task)\b[:\-]?", "", task).strip(" -:•")
            if task:
                action_items.append({"task": task, "owner": owner, "due": due})
    action_items = action_items[:10]

    decisions = []
    for line in ai_text.splitlines():
        if _re.search(r"(?i)\b(decision|decided)\b", line):
            decisions.append(line.strip())
    decisions = decisions[:10]

    # Sentiment guess
    sent = "Unknown"
    if _re.search(r"(?i)\bpositive\b", ai_text): sent = "Positive"
    elif _re.search(r"(?i)\bneutral\b", ai_text): sent = "Neutral"
    elif _re.search(r"(?i)\bnegative\b", ai_text): sent = "Negative"

    return {
        "executive_summary": exec_sum[:2000],
        "key_points": key_points,
        "action_items": action_items,
        "decisions": decisions,
        "sentiment": sent
    }


def summarize_text(transcript: str, target_language: str = "auto") -> dict:
    """
    Summarize transcript into structured meeting notes.
    If target_language == "auto", we default summaries to English.
    Else we summarize in the selected language.

    The model is asked to return STRICT JSON only, so we can render bullets and action items properly.
    """
    lang_name = _code_to_lang_name(target_language)

    prompt = f"""
You are a professional AI meeting assistant.
Summarize the following transcript into structured notes in {lang_name}.

Return ONLY valid JSON (no markdown, no code fences, no extra text) with this exact schema:
{{
  "executive_summary": "string, 3-6 sentences",
  "key_points": ["bullet 1", "bullet 2", "... (max 10)"],
  "action_items": [
    {{"task": "what needs to be done", "owner": "person or empty", "due": "date or empty"}}
  ],
  "decisions": ["decision 1", "... (max 10)"],
  "sentiment": "Positive" | "Neutral" | "Negative"
}}

Guidelines:
- Be concise and factually accurate (aim 90%+).
- Extract explicit owners/dates if mentioned; otherwise use empty strings.
- Do not include any keys other than the schema.
- Do not add explanations outside JSON.

Transcript:
{transcript[:12000]}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise and efficient summarizer that outputs strict JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # lower temp for consistency/accuracy
        )

        ai_text = ""
        if getattr(response, "choices", None):
            msg = response.choices[0].message
            ai_text = getattr(msg, "content", "") or ""

        structured = _coerce_structured_notes(ai_text, transcript)
        log.info("Summary structured (preview): %s", str(structured)[:200])
        return structured

    except Exception as e:
        log.exception("Summarization failed: %s", e)
        return {
            "executive_summary": transcript[:400] + ("..." if len(transcript) > 400 else ""),
            "key_points": [],
            "action_items": [],
            "decisions": [],
            "sentiment": "Unknown"
        }
