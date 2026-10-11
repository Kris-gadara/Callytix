import os
import shutil
import logging
import tempfile
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from train import train_model, MODEL_PATH, VECTORIZER_PATH
from predict import predict_call
from whisper_service import transcribe_audio

logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

app = FastAPI(title="Callytix - AI-Based Fraud Phone Call Detection System")
MAX_AUDIO_BYTES = 100 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Set up templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
async def read_home(request: Request):
    """
    Renders the homepage (index.html).
    Passes information on whether the model is already trained.
    """
    is_trained = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"is_trained": is_trained}
    )

@app.get("/healthz")
async def health_check():
    """Reports whether required prediction assets and FFmpeg are available."""
    model_ready = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
    ffmpeg_ready = shutil.which("ffmpeg") is not None
    if not model_ready or not ffmpeg_ready:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "model_ready": model_ready,
                "ffmpeg_ready": ffmpeg_ready,
            },
        )
    return {"status": "ok"}

@app.post("/train")
async def handle_train():
    """
    Trains the TF-IDF Vectorizer and Multinomial Naive Bayes model
    on the Kaggle dataset. Returns the evaluation metrics as JSON.
    """
    try:
        metrics = await run_in_threadpool(train_model)
        return JSONResponse(content={
            "success": True,
            "message": "Model trained successfully!",
            "metrics": metrics
        })
    except Exception as e:
        logger.exception("Model training failed")
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": f"Training failed: {str(e)}"
        })

@app.post("/predict", response_class=HTMLResponse)
async def handle_predict(request: Request, audio_file: UploadFile = File(...)):
    """
    Handles audio file uploads, transcribes them with Whisper,
    runs the fraud prediction pipeline, and renders the results page.
    """
    # 1. Validate that the model is trained
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        # Render index page with error message
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "is_trained": False,
                "error": "The model has not been trained yet. Please train the model before predicting."
            },
            status_code=503,
        )
        
    temp_file_path = None
    try:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No file was selected for upload.")

        filename = os.path.basename(audio_file.filename.replace("\\", "/"))
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in {".wav", ".mp3", ".m4a", ".flac"}:
            raise HTTPException(
                status_code=415,
                detail="Unsupported audio format. Upload a WAV, MP3, M4A, or FLAC file.",
            )

        with tempfile.NamedTemporaryFile(
            mode="wb", dir=UPLOAD_DIR, suffix=suffix, delete=False
        ) as temporary_file:
            temp_file_path = temporary_file.name
            total_bytes = 0
            while chunk := await audio_file.read(UPLOAD_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > MAX_AUDIO_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="The audio file exceeds the 100 MB upload limit.",
                    )
                temporary_file.write(chunk)

        transcript = await run_in_threadpool(transcribe_audio, temp_file_path)
        prediction_result = await run_in_threadpool(predict_call, transcript)

        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "filename": filename,
                "transcript": transcript if transcript else "[No speech detected in audio file]",
                "prediction": prediction_result["prediction"],
                "confidence": prediction_result["confidence"],
                "detected_keywords": prediction_result["detected_keywords"]
            }
        )

    except HTTPException as e:
        if e.status_code >= 500:
            logger.exception("Prediction request failed")
        is_trained = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"is_trained": is_trained, "error": e.detail},
            status_code=e.status_code,
        )
    except Exception:
        logger.exception("Prediction request failed")
        is_trained = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "is_trained": is_trained,
                "error": "The audio could not be analyzed. Check the service logs for details and try again."
            },
            status_code=500,
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        await audio_file.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)
