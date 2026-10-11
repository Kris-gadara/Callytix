import os
import shutil
import tempfile
import threading

import imageio_ffmpeg
import torch
import whisper

WHISPER_MODEL = "small"
_model = None
_model_lock = threading.Lock()
_transcription_lock = threading.Lock()


def is_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None or os.path.isfile(
        imageio_ffmpeg.get_ffmpeg_exe()
    )


def _ensure_ffmpeg_available():
    if shutil.which("ffmpeg") is not None:
        return
    bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_directory = os.path.join(tempfile.gettempdir(), "callytix-ffmpeg")
    executable_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    ffmpeg_path = os.path.join(ffmpeg_directory, executable_name)
    os.makedirs(ffmpeg_directory, exist_ok=True)
    if not os.path.exists(ffmpeg_path):
        try:
            os.link(bundled_ffmpeg, ffmpeg_path)
        except FileExistsError:
            pass
        except OSError:
            shutil.copy2(bundled_ffmpeg, ffmpeg_path)

    os.environ["PATH"] = (
        ffmpeg_directory + os.pathsep + os.environ.get("PATH", "")
    )
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("The bundled FFmpeg executable could not be started.")


def get_whisper_model():
    """Load and cache the local Whisper model."""
    global _model
    with _model_lock:
        if _model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading Whisper '{WHISPER_MODEL}' model on {device}...")
            _model = whisper.load_model(WHISPER_MODEL, device=device)
            print("Whisper model loaded successfully.")
    return _model


def transcribe_audio(audio_path: str) -> str:
    """Transcribe a local audio file with OpenAI Whisper."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    with _transcription_lock:
        _ensure_ffmpeg_available()
        result = get_whisper_model().transcribe(audio_path)
        return result.get("text", "").strip()
