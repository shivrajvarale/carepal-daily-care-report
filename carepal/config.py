
import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_TRANSCRIBE_MODEL = os.getenv("GEMINI_TRANSCRIBE_MODEL", GEMINI_MODEL)
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
MAX_NOTE_CHARS = int(os.getenv("MAX_NOTE_CHARS", "6000"))
MAX_AUDIO_MB = int(os.getenv("MAX_AUDIO_MB", "10"))

SUMMARY_LANGUAGES = {
    "English": "English",
    "Hindi": "Hindi (Devanagari script)",
    "Marathi": "Marathi (Devanagari script)",
    "Hinglish": "Hinglish (Hindi written in Roman script, the way people text)",
}
