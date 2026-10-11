# Callytix: AI-Based Fraud Phone Call Detection

Callytix analyzes an uploaded phone-call recording locally. OpenAI Whisper
transcribes the audio, then a saved TF-IDF vectorizer and Multinomial Naive
Bayes classifier predict whether the transcript appears fraudulent or
legitimate. The result page also shows confidence and suspicious keywords;
keywords are informational and do not determine the prediction.

## Requirements

- Python 3.10 or newer
- The Python dependencies in `requirements.txt`; `imageio-ffmpeg` supplies
  Whisper's FFmpeg executable, so a separate system FFmpeg install is optional.

## Install and run on Windows

Run these commands from the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:8000>. After the one-time setup, start the app in a new
terminal with:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

If PowerShell blocks virtual-environment activation, use
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt` to install and
`.\.venv\Scripts\python.exe app.py` to run without activating it.

## Install and run on macOS or Linux

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:8000>.

## Using Callytix

1. Open the local app and upload a WAV, MP3, M4A, or FLAC recording.
2. Submit the form to transcribe the call and display its classification.
3. The included `models/model.pkl` and `models/vectorizer.pkl` are used for
   predictions; they do not need to be retrained for normal use.
4. Use **Train Model Now** or run `python train.py` to download the public
   Kaggle transcript dataset, retrain the classifier, and save updated model
   files under `models/`.

Whisper downloads its `small` model the first time it transcribes audio. Later
runs use the locally cached model. Uploaded files are temporary and removed
after analysis.

## Project structure

```text
Callytix/
├── app.py                  # Local FastAPI entry point (127.0.0.1:8000)
├── predict.py              # Loads model artifacts and classifies transcripts
├── preprocess.py           # Text cleaning and stopword removal
├── train.py                # Kaggle dataset training and evaluation pipeline
├── whisper_service.py      # Local Whisper model loading and transcription
├── requirements.txt        # Python dependencies
├── README.md               # Local setup and usage
├── models/
│   ├── model.pkl           # Trained Naive Bayes classifier
│   └── vectorizer.pkl      # Fitted TF-IDF vectorizer
├── templates/
│   ├── index.html          # Audio upload and training interface
│   └── result.html         # Prediction results
├── static/
│   ├── script.js           # Upload and training interface behavior
│   └── style.css           # Local interface styles
├── uploads/                # Temporary audio uploads during analysis
└── demo_testing.png        # Project screenshot
```
