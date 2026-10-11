---
title: Callytix Fraud Call Detection
emoji: 📞
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.49.1"
python_version: "3.12.12"
app_file: app.py
pinned: false
---

# Callytix: AI-Based Fraud Phone Call Detection System

Callytix is an AI-based project designed to detect fraudulent phone calls (such as bank impersonations, lottery scams, and OTP harvesting) from audio recordings. The system transcribes the call audio using OpenAI Whisper and classifies the text as **Fraudulent** or **Legitimate** using a Multinomial Naive Bayes classifier trained on a composite scam transcript dataset.

## Hugging Face Spaces

This repository is configured for the Gradio SDK. To use ZeroGPU for speech
transcription, select **ZeroGPU** in the Space's hardware settings. ZeroGPU
hosting eligibility and daily GPU quotas are controlled by Hugging Face; this
repository configuration does not grant access to that hardware.

The Space uses the included `models/model.pkl` and `models/vectorizer.pkl`
artifacts. If they are missing, train from the **Train classifier** tab. A
training run saves files only to the current Space runtime; commit the updated
model artifacts to the repository if they need to remain available after a
restart. Whisper uses the `tiny` model by default to limit the first model
download and runtime use. Set `WHISPER_MODEL` in Space variables to choose a
larger supported model. `packages.txt` installs FFmpeg, which Whisper requires
to decode audio.

---

## 🚀 Features

- **Speech-to-Text Transcription**: Automatically transcribes uploaded audio files (`.wav`, `.mp3`, `.m4a`, etc.) using the OpenAI Whisper `tiny` model by default.
- **Machine Learning Classification**: Predicts whether the transcribed text is Fraudulent or Legitimate using TF-IDF vectorization and a Multinomial Naive Bayes model.
- **Confidence Scoring**: Computes and displays the model's confidence percentage for the prediction.
- **Explainable Highlights**: Shows suspicious terms (like `OTP`, `KYC`, `Bank`, `Customs`) found in the transcript. These terms do not decide the classification.
- **On-Demand Training**: Allows training and evaluating the model directly from the web interface, displaying validation metrics (Accuracy, Precision, Recall, F1 Score) and a Confusion Matrix.
- **College-Focused Architecture**: Structured, clean, and beginner-friendly codebase containing no databases, complex microservices, or external API costs.

---

## 📊 Dataset

The project uses the **Composite Scam Transcript Dataset** hosted on Kaggle (`ibrahimbagwan12/composite-scam-transcript-dataset`).
- **Total Samples**: ~36,750 training examples and ~9,180 testing examples.
- **Columns**:
  - `text`: Text transcript of the conversation.
  - `label`: Binary classification (1 = Fraudulent/Scam, 0 = Legitimate/Normal).

---

## ⚙️ Project Architecture

```
                      +-------------------+
                      |   Audio Upload    |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   OpenAI Whisper  |
                      |   (tiny model)    |
                      +---------+---------+
                                | (Transcript Text)
                                v
                      +-------------------+
                      |   Text Cleaning   |
                      |  (preprocess.py)  |
                      +---------+---------+
                                | (Clean Tokens)
                                v
                      +-------------------+
                      | TF-IDF Vectorizer |
                      | (vectorizer.pkl)  |
                      +---------+---------+
                                | (Feature Vector)
                                v
                      +-------------------+
                      |  Multinomial NB   |
                      |    (model.pkl)    |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      | Prediction Output |
                      |    (Confidence)   |
                      +-------------------+
```

---

## 📂 Folder Structure

```text
Callytix/
│
├── app.py                  # Gradio interface and Space entry point
├── train.py                # Machine Learning Training Pipeline
├── predict.py              # ML Prediction & Suspicious Keyword Extraction
├── whisper_service.py      # Speech-to-Text via OpenAI Whisper
├── preprocess.py           # Text Preprocessing & Stopword Removal
├── requirements.txt        # Gradio Space Python dependencies
├── packages.txt            # System packages required by Hugging Face Spaces
├── README.md               # Project Documentation
│
├── models/
│   ├── model.pkl           # Saved Naive Bayes Model weights
│   └── vectorizer.pkl      # Saved TF-IDF Vectorizer weights
│
├── templates/
│   ├── index.html          # Earlier HTML interface (not used by Gradio)
│   └── result.html         # Earlier results template (not used by Gradio)
│
└── static/
    ├── style.css           # Earlier interface styles (not used by Gradio)
    └── script.js           # Earlier interface behavior (not used by Gradio)
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** and **pip** installed.
Additionally, OpenAI Whisper requires **FFmpeg** to process and decode audio files.

#### Windows FFmpeg Setup (Automated via winget)
The project automatically checks for a Windows package manager (`winget`) installation of FFmpeg at runtime. To install it:
1. Open PowerShell or Command Prompt.
2. Run the following command:
   ```bash
   winget install Gyan.FFmpeg.Essentials --accept-source-agreements --accept-package-agreements
   ```
3. The backend script `whisper_service.py` is configured to dynamically find this WinGet installation and append it to your environment `PATH` at runtime, so no system restart or manual variable setup is needed!

#### macOS & Linux FFmpeg Setup
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

### 2. One-time Setup
1. Clone or extract the project files to a local directory.
2. Create and activate a virtual environment, then install the dependencies once. To prevent compatibility issues between precompiled PyTorch binaries and NumPy 2.x, the project pins `numpy<2.0.0`:
   ```bash
   python -m venv .venv
   ```

   On Windows PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements-render.txt
   ```

   On macOS/Linux:
   ```bash
   source .venv/bin/activate
   python -m pip install -r requirements-render.txt
   ```

   Keep the virtual environment activated in your project terminal. Dependencies only need to be installed again if you recreate `.venv` or change the requirements.

---

## 🏋️ Training the Model

Before making predictions, you must train the machine learning model. You can do this in two ways:

### Option A: Command Line (Recommended first run)
Run the training script standalone:
```bash
python train.py
```
This will:
1. Download the Kaggle dataset automatically via `kagglehub`.
2. Preprocess the transcripts.
3. Train the model.
4. Print the final metrics (Accuracy, Precision, Recall, F1 Score) and Confusion Matrix.
5. Save the trained outputs inside the `models/` directory.

### Option B: Gradio interface
Start the web app and click **"Train model"** in the **Train classifier** tab.

---

## 🖥️ Running the Application

To start the Gradio web interface:
```bash
python app.py
```

The app listens on port `7860` by default. To use a different port, set `PORT` before starting it. Once started, open:
```
http://127.0.0.1:7860/
```

1. Select or drag a call audio recording (wav/mp3) into the uploader.
2. Click **"Transcribe and analyze"**.
3. View the transcript, the classification output (Fraudulent vs. Legitimate), the confidence score, and any suspicious keywords.

### Render Deployment

The repository includes a Render Blueprint (`render.yaml`), a Linux FFmpeg package list (`apt.txt`), and Render-specific requirements that install CPU-only PyTorch (`requirements-render.txt`) to avoid downloading CUDA libraries. Render is configured to use the Whisper `tiny` model by default. Set `WHISPER_MODEL` in the Render environment to `base` or `small` if more transcription accuracy is needed and the instance has enough memory. The application reads this variable when loading the model.

Deploy the Blueprint from the repository root, or configure an existing Render web service with:

- **Build command:** `pip install -r requirements-render.txt`
- **Start command:** `python app.py`
- **Health check path:** `/`

The Whisper model may need to download its weights at startup, so the first startup or prediction can take longer. If a prediction returns `502 Bad Gateway`, inspect the Render logs for memory exhaustion or a worker restart.

---

## 🔧 Troubleshooting

If you encounter errors during setup or prediction, check these solutions:

### 1. `Numpy is not available` / `Failed to initialize NumPy: _ARRAY_API not found`
- **Cause**: PyTorch was compiled with NumPy 1.x, but your environment has NumPy 2.x installed.
- **Solution**: Downgrade NumPy to 1.x:
  ```bash
  pip install "numpy<2.0.0"
  ```
  The `requirements.txt` file has been pinned to `numpy<2.0.0` to avoid this.

### 2. `Speech-to-text transcription failed: [WinError 2] The system cannot find the file specified`
- **Cause**: OpenAI Whisper cannot locate the `ffmpeg` executable in your system execution `PATH`.
- **Solution**: Install FFmpeg using `winget` as specified in the installation steps. The app will automatically resolve and inject the path at runtime. If you installed FFmpeg manually elsewhere, ensure its `bin` folder is added to your User/System environment variables.

---

## 📸 Screenshots

### System UI (Main Dashboard)
Here is the homepage interface for uploading call recordings and managing the machine learning model:

![System UI](system_ui.png)

### Demo Testing (Analysis Report)
Here is the result screen showing the transcribed text, model classification prediction, confidence score, and highlighted keywords:

![Demo Testing](demo_testing.png)


---

## Information

1. **Why use Multinomial Naive Bayes instead of Deep Learning (like BERT)?**
   - Naive Bayes is extremely fast, highly efficient, and uses very little memory. It works exceptionally well for text classification when combined with TF-IDF, making it easy to run on standard computers without needing an expensive GPU. It serves as an excellent baseline and keeps the project code simple and easy to explain.

2. **How does TF-IDF work?**
   - TF-IDF stands for *Term Frequency-Inverse Document Frequency*. It converts text into numerical vectors. 
     - **Term Frequency (TF)** measures how often a word appears in a specific document.
     - **Inverse Document Frequency (IDF)** measures how unique/rare the word is across all documents in the dataset. Words like "the" get low scores, while words like "OTP" or "verification" get higher scores because they are rare and carry more information.

3. **What is the Naive Bayes assumption?**
   - It assumes that the presence of a particular feature (word) in a class is completely independent of the presence of any other feature. Although this assumption is "naive" and rarely holds true in natural language, Naive Bayes classifiers still perform remarkably well in practice.

4. **Why use Whisper tiny model by default?**
   - The `tiny` model reduces initial download time and resource use for free hosting. Larger supported models can be selected with the `WHISPER_MODEL` environment variable if the available hardware and quota permit.

---

## 🔮 Future Scope

- **Real-Time Call Interception**: Developing a mobile plugin or VoIP gateway to analyze incoming calls in real-time.
- **Multi-Lingual Support**: Extending training datasets to include regional/vernacular languages (e.g., Hindi, Spanish) where fraud is highly active.
- **Voice Biometrics**: Analyzing caller pitch, stress levels, and background noise anomalies to detect synthesized or deepfake voice profiles.
