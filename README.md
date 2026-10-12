# Callytix: AI-Based Fraud Phone Call Detection

Callytix is a local web application for reviewing potentially fraudulent phone
calls. A user uploads a call recording, the app transcribes it with OpenAI
Whisper, and a saved text-classification pipeline predicts whether the
transcript resembles a fraudulent or legitimate call. The result includes the
transcript, model confidence, and matching suspicious keywords.

The web interface is inference-only: users can upload audio and view a
prediction. It does not expose model training. The trained model and fitted
vectorizer are loaded from `models/` and are used as-is.

## Technology

| Area | Technology |
| --- | --- |
| Web application and HTTP routes | Python, FastAPI, Uvicorn |
| HTML rendering | Jinja2 templates |
| Front end | HTML, CSS, and browser JavaScript |
| Audio transcription | OpenAI Whisper (`small` model), PyTorch |
| Audio decoding | FFmpeg, with `imageio-ffmpeg` as a bundled fallback |
| Text classification | scikit-learn TF-IDF vectorizer and Multinomial Naive Bayes |
| Model serialization | Joblib (`model.pkl` and `vectorizer.pkl`) |
| Optional model training | pandas, scikit-learn, and KaggleHub |

## Model and machine-learning method

This is a **supervised binary text-classification** project. It does not
classify raw audio directly: Whisper first converts speech into transcript
text, and the text model classifies that transcript.

- **Dataset:** the Kaggle Composite Scam Transcript Dataset, file
  `composite_train.csv`, loaded by `train.py`. The expected columns are `text`
  and `label`.
- **Labels:** `0` represents legitimate and `1` represents fraudulent/scam,
  as assumed by `predict.py`.
- **Preprocessing:** `preprocess.py` lowercases text, removes digits,
  punctuation and other non-English-letter characters, splits into words, and
  removes its built-in English stopword list.
- **Text features:** scikit-learn `TfidfVectorizer(max_features=5000)` learns
  up to 5,000 unigram (single-word) features from the training subset. TF-IDF
  gives higher weight to terms that are informative within a document and
  less common across the corpus.
- **Classifier:** scikit-learn `MultinomialNB` learns class-specific word
  statistics from the sparse TF-IDF vectors. Its default additive smoothing
  parameter is `alpha=1.0`.
- **Saved artifacts:** `models/vectorizer.pkl` and `models/model.pkl` are a
  matched pair. The running app loads these artifacts and performs inference;
  it does not fit or update them.

The checked-in model artifact currently reports 5,000 features and was fitted
with 29,385 training examples: 12,770 label-0 examples and 16,615 label-1
examples. These are model artifact metadata, not a claim about the evaluation
score. Training metrics are printed when `train.py` is run and are not
currently persisted as a report.

The percentage returned by `predict_proba` is the model's estimated class
probability; it is not a separately calibrated real-world risk score.
Multinomial Naive Bayes with unigram TF-IDF does not understand word order,
sarcasm, or conversational context, and transcription mistakes can affect the
result. A transcript with no usable words currently falls back to
`Legitimate` at 50%; this should be treated as **unable to assess**, not as
evidence that the call is safe.

## Training, test split, and examples

When a maintainer runs `python train.py`, the training process:

1. Downloads/loads `composite_train.csv` from KaggleHub.
2. Drops rows with missing text or labels; cleans each transcript; drops
   transcripts that become empty; converts labels to integers.
3. Splits the cleaned labeled data with
   `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
   This creates an 80% training subset and a held-out 20% test subset.
   `stratify=y` keeps the class proportions approximately consistent in both
   subsets, and the fixed seed makes the split repeatable for the same input
   data.
4. Fits the TF-IDF vocabulary and weights on **training text only**; uses that
   fitted vectorizer to transform both the training and held-out test text.
5. Fits Multinomial Naive Bayes on the training vectors and labels.
6. Predicts the held-out test labels and prints accuracy, precision, recall,
   F1 score, and a confusion matrix.
7. Saves the fitted classifier and vectorizer into `models/`.

Illustrative rows (examples only; they are not quoted rows from the Kaggle
dataset):

| Transcript text | Example label |
| --- | --- |
| “Your appointment is confirmed for Thursday at three.” | `0` — legitimate |
| “Share your bank OTP now to stop your account from being blocked.” | `1` — scam |

After preprocessing, words such as `your`, `is`, `for`, and `at` are removed,
and numbers and punctuation are discarded. The remaining text is transformed
using the vocabulary learned from the training subset. During evaluation, the
test subset is not used to fit the vocabulary or classifier; it is used only
to measure how the fitted pipeline predicts unseen rows from that split.

As a separate inference demonstration, the current saved artifacts classify
the first example above as **Legitimate (62.55%)**. They classify “Share your
bank OTP now to stop your account from being blocked” as **Fraudulent
(85.00%)**, with `OTP`, `Bank`, and `Account` highlighted. These are example
predictions on manually supplied sentences, not test-set metrics or quotes
from the dataset.

The evaluation metrics treat class `1` (scam) as the positive class:

- **Accuracy:** correct predictions divided by all predictions.
- **Precision:** of calls predicted as scams, the fraction that are labeled
  scams (`TP / (TP + FP)`).
- **Recall:** of the labeled scams, the fraction detected (`TP / (TP + FN)`).
- **F1:** the harmonic mean of precision and recall; useful when both missed
  scams and false alarms matter.
- **Confusion matrix:** counts true negatives, false positives, false
  negatives, and true positives. The code uses scikit-learn's label order
  `[0, 1]`, so rows are actual labels and columns are predicted labels.

For example, a **hypothetical** test result with `TP=80`, `FP=20`, `FN=10`,
and `TN=90` would have 85% accuracy, 80% precision, about 88.9% recall, and
about 84.2% F1. This only demonstrates the metric calculations; it is not a
reported Callytix model result. For a project evaluation, run the training
script and record the actual output with the dataset version used.

## End-to-end flow

1. **Start the local server.** `app.py` starts FastAPI and serves the home page,
   static assets, and the health endpoint.
2. **Check model readiness.** The home page checks for both saved model
   artifacts. When they are present, it shows the audio upload form.
3. **Upload a recording.** The user selects or drops a WAV, MP3, M4A, or FLAC
   file. The server validates its extension and enforces a 100 MB upload limit.
4. **Transcribe the audio.** `whisper_service.py` makes FFmpeg available,
   loads and caches Whisper locally, and converts speech to transcript text.
5. **Classify the transcript.** `predict.py` cleans the text with
   `preprocess.py`, transforms it with the saved TF-IDF vectorizer, and asks the
   saved Multinomial Naive Bayes model for a class and probability.
6. **Show the result.** The result page displays the prediction, confidence,
   transcript, and any matching suspicious keywords. Keywords are explanatory
   only; they do not determine the model's prediction.
7. **Clean up the upload.** The temporary audio file is removed after the
   request completes, including when analysis returns an error.

The classifier output is informational, not a guarantee that a call is safe or
fraudulent. Review the transcript and use independent judgment.

## Requirements

- Python 3.10 or newer
- Dependencies listed in `requirements.txt`
- The included trained files: `models/model.pkl` and
  `models/vectorizer.pkl`

Whisper downloads its `small` model the first time it transcribes audio. The
model is cached locally for later runs. `imageio-ffmpeg` supplies FFmpeg when a
system FFmpeg executable is not installed.

## Install and run on Windows

Run these commands from the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:8000>. To start the app later, activate the environment
and run `python app.py` again. If PowerShell blocks activation, use
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and
`.\.venv\Scripts\python.exe app.py`.

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

## Routes

- `GET /` — prediction-only audio upload page, with model readiness status.
- `POST /predict` — accepts an audio upload and renders the analysis result.
- `GET /healthz` — returns service readiness, including model files and FFmpeg.

## Optional maintainer model training

Training is not available from the web app. A maintainer can run
`python train.py` separately to download the Composite Scam Transcript Dataset
through KaggleHub, clean and split the labeled transcripts, fit a TF-IDF
vectorizer and Multinomial Naive Bayes classifier, evaluate the model, and
replace the files in `models/`. The web app uses those saved artifacts only;
training dependencies are not required by the prediction request flow.

## Project structure

```text
Callytix/
├── app.py                  # FastAPI server, upload validation, and prediction route
├── predict.py              # Loads saved assets and classifies transcripts
├── preprocess.py           # Text normalization and stopword removal
├── whisper_service.py      # Local Whisper transcription and FFmpeg setup
├── train.py                # Optional, standalone model training utility
├── requirements.txt        # Python dependencies
├── models/
│   ├── model.pkl           # Trained Naive Bayes classifier
│   └── vectorizer.pkl      # Fitted TF-IDF vectorizer
├── templates/
│   ├── index.html          # Prediction-only audio upload page
│   └── result.html         # Prediction report and transcript
├── static/
│   ├── script.js           # Upload, drag-and-drop, and loading behavior
│   └── style.css           # Responsive page and result styling
└── uploads/                # Temporary analysis uploads; removed after processing
```
