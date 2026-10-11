import logging
import os

import gradio as gr

from predict import MODEL_PATH, VECTORIZER_PATH, predict_call
from train import train_model
from whisper_service import transcribe_audio

logger = logging.getLogger(__name__)
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a", ".flac"}
MAX_AUDIO_BYTES = 100 * 1024 * 1024


def analyze_audio(audio_path):
    if not audio_path:
        raise gr.Error("Upload an audio file before starting analysis.")

    suffix = os.path.splitext(audio_path)[1].lower()
    if suffix not in SUPPORTED_AUDIO_FORMATS:
        raise gr.Error("Unsupported audio format. Upload a WAV, MP3, M4A, or FLAC file.")

    if os.path.getsize(audio_path) > MAX_AUDIO_BYTES:
        raise gr.Error("The audio file exceeds the 100 MB upload limit.")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise gr.Error("The classifier is not trained yet. Use the training button first.")

    try:
        transcript = transcribe_audio(audio_path)
        result = predict_call(transcript)
    except Exception as exc:
        logger.exception("Audio analysis failed")
        raise gr.Error(f"Audio analysis failed: {exc}") from exc

    transcript = transcript or "[No speech detected in audio file]"
    keywords = ", ".join(result["detected_keywords"]) or "None detected"
    return (
        transcript,
        result["prediction"],
        f'{result["confidence"]:.2f}%',
        keywords,
    )


def train_classifier():
    try:
        metrics = train_model()
    except Exception as exc:
        logger.exception("Model training failed")
        raise gr.Error(f"Model training failed: {exc}") from exc

    matrix = metrics["confusion_matrix"]
    return (
        "### Training complete\n"
        f'- Accuracy: **{metrics["accuracy"]:.2%}**\n'
        f'- Precision: **{metrics["precision"]:.2%}**\n'
        f'- Recall: **{metrics["recall"]:.2%}**\n'
        f'- F1 score: **{metrics["f1_score"]:.2%}**\n\n'
        "Confusion matrix (actual rows, predicted columns):\n\n"
        f"| | Predicted legitimate | Predicted fraudulent |\n"
        f"|---|---:|---:|\n"
        f"| Actual legitimate | {matrix[0][0]} | {matrix[0][1]} |\n"
        f"| Actual fraudulent | {matrix[1][0]} | {matrix[1][1]} |"
    )


with gr.Blocks(title="Callytix - Fraud Call Detection") as demo:
    gr.Markdown(
        "# Callytix: AI-Based Fraud Phone Call Detection\n"
        "Upload a call recording. Whisper transcribes it; the trained "
        "TF-IDF and Multinomial Naive Bayes classifier predicts whether "
        "the transcript appears fraudulent or legitimate."
    )

    with gr.Tab("Analyze a call"):
        gr.Markdown(
            "> If the saved classifier files are missing, train the model in the "
            "**Train classifier** tab before analyzing calls."
        )

        audio_input = gr.Audio(
            sources=["upload"],
            type="filepath",
            label="Call recording (WAV, MP3, M4A, or FLAC; max 100 MB)",
        )
        analyze_button = gr.Button("Transcribe and analyze", variant="primary")
        prediction_output = gr.Textbox(label="Prediction")
        confidence_output = gr.Textbox(label="Classifier confidence")
        keywords_output = gr.Textbox(label="Suspicious keywords (informational only)")
        transcript_output = gr.Textbox(label="Transcript", lines=8)
        analyze_button.click(
            fn=analyze_audio,
            inputs=audio_input,
            outputs=[
                transcript_output,
                prediction_output,
                confidence_output,
                keywords_output,
            ],
            concurrency_limit=1,
        )

    with gr.Tab("Train classifier"):
        gr.Markdown(
            "Downloads the public Kaggle transcript dataset, trains the "
            "TF-IDF and Multinomial Naive Bayes classifier, evaluates a held-out "
            "20% split, and saves the model files in this Space's runtime."
        )
        train_button = gr.Button("Train model", variant="secondary")
        training_output = gr.Markdown()
        train_button.click(fn=train_classifier, outputs=training_output)

    gr.Markdown(
        "**Note:** Keyword matches are shown as clues only and do not determine "
        "the classifier's prediction. Predictions are not a guarantee that a "
        "call is safe or fraudulent."
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )
