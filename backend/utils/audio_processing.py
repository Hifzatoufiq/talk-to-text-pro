import os
import logging
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------- OpenAI (HARDCODED KEY) ----------
OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)

# Absolute uploads path (mirror app.py)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_audio_from_file(filepath: str) -> str:
    """
    Convert any input (audio or video) to a mono 16k WAV.
    1) Try pydub (ffmpeg required), fallback to moviepy.
    """
    filename = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(UPLOAD_FOLDER, filename + "_converted.wav")
    log.info("Extracting audio from: %s -> %s", filepath, out_path)

    # Try pydub first
    try:
        audio = AudioSegment.from_file(filepath)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(out_path, format="wav")
        log.info("Audio extracted with pydub to %s", out_path)
        return out_path
    except Exception as e:
        log.warning("pydub failed, trying moviepy: %s", e)

    # Fallback: moviepy
    try:
        clip = VideoFileClip(filepath)
        if clip.audio is None:
            clip.close()
            raise ValueError("No audio stream in media")
        clip.audio.write_audiofile(out_path, fps=16000, codec="pcm_s16le")
        clip.close()
        log.info("Audio extracted with moviepy to %s", out_path)
        return out_path
    except Exception as e:
        log.error("Failed to extract audio: %s", e)
        raise e

def transcribe_audio(filepath: str, language: str = "auto") -> str:
    """
    Transcribe with OpenAI Whisper (whisper-1).
    Returns text (empty string on failure).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    try:
        kwargs = {"model": "whisper-1"}
        if language and language.lower() != "auto":
            kwargs["language"] = language  # hint Whisper

        with open(filepath, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(file=audio_file, **kwargs)

        text = getattr(transcript, "text", None) or str(transcript)
        log.info("Transcription success (preview): %s", (text[:120] + "...") if len(text) > 120 else text)
        return text
    except Exception as e:
        log.exception("Whisper transcription failed: %s", e)
        return ""

def _code_to_lang_name(code: str) -> str:
    mapping = {"en": "English", "ur": "Urdu", "hi": "Hindi", "auto": "Auto"}
    return mapping.get((code or "").lower(), "Auto")

def summarize_text(transcript: str, target_language: str = "auto") -> dict:
    """
    Summarize transcript into structured meeting notes.
    """
    lang_name = _code_to_lang_name(target_language)
    write_instr = (
        "Write all sections in the same language as the transcript."
        if lang_name == "Auto"
        else f"Write all sections in {lang_name}."
    )

    prompt = f"""
You are an AI meeting assistant. Summarize the following transcript into structured meeting notes.
{write_instr}

Include these sections:

- Executive Summary
- Key Points (bullet list)
- Action Items (task, owner, due date if mentioned)
- Decisions
- Sentiment (Positive, Neutral, Negative)

Transcript:
{transcript}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful meeting assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        ai_output = None
        if hasattr(response, "choices") and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                ai_output = choice.message.content
            elif hasattr(choice, "text"):
                ai_output = choice.text

        if ai_output is None:
            ai_output = str(response)

        log.info("Summary generated (preview): %s", (ai_output[:120] + "...") if len(ai_output) > 120 else ai_output)

        return {
            "executive_summary": ai_output,
            "key_points": [],
            "action_items": [],
            "decisions": [],
            "sentiment": "Unknown"
        }
    except Exception as e:
        log.exception("OpenAI summarization failed: %s", e)
        return {
            "executive_summary": transcript[:400] + ("..." if len(transcript) > 400 else ""),
            "key_points": [],
            "action_items": [],
            "decisions": [],
            "sentiment": "Unknown"
        }
