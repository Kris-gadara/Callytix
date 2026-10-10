import whisper
import torch
import os
import threading

# Add winget FFmpeg paths dynamically to PATH environment variable
# to ensure Whisper can locate ffmpeg without needing shell/machine restart.
def _setup_ffmpeg_path():
    user_profile = os.environ.get("USERPROFILE", "")
    winget_paths = [
        os.path.join(user_profile, r"AppData\Local\Microsoft\WinGet\Links"),
        os.path.join(user_profile, r"AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-essentials_build\bin")
    ]
    current_path = os.environ.get("PATH", "")
    new_paths = [p for p in winget_paths if os.path.exists(p) and p not in current_path]
    if new_paths:
        os.environ["PATH"] = ";".join(new_paths) + ";" + current_path
        print(f"Dynamically added FFmpeg to PATH: {new_paths}")

_setup_ffmpeg_path()

# Global variable to cache the whisper model so we don't reload it on every prediction
_model = None
_model_lock = threading.Lock()
_transcription_lock = threading.Lock()

def get_whisper_model():
    """
    Loads and returns the Whisper 'small' model.
    Caches the model to optimize prediction speed.
    """
    global _model
    with _model_lock:
        if _model is None:
            print("Loading Whisper 'small' model (this may take a moment on first run)...")
            # Check if CUDA (GPU) is available, otherwise default to CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Whisper will run on device: {device}")

            _model = whisper.load_model("small", device=device)
            print("Whisper model loaded successfully!")
    return _model

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes the audio file at the given path using Whisper.
    Returns the transcription text.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        
    with _transcription_lock:
        model = get_whisper_model()
        print(f"Transcribing audio file: {audio_path}...")

        result = model.transcribe(audio_path)

        transcription = result.get("text", "").strip()
        print("Transcription complete!")
        return transcription
