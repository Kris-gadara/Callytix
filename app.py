import logging
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from predict import MODEL_PATH, VECTORIZER_PATH, predict_call
from train import train_model
from whisper_service import is_ffmpeg_available, transcribe_audio

logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
MAX_AUDIO_BYTES = 100 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a", ".flac"}

os.makedirs(UPLOADS_DIR, exist_ok=True)

app = FastAPI(title="Callytix - AI-Based Fraud Phone Call Detection System")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def read_home(request: Request):
    is_trained = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"is_trained": is_trained},
    )


@app.get("/healthz")
async def health_check():
    model_ready = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
    ffmpeg_ready = is_ffmpeg_available()
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
    try:
        metrics = await run_in_threadpool(train_model)
        return {
            "success": True,
            "message": "Model trained successfully!",
            "metrics": metrics,
        }
    except Exception as exc:
        logger.exception("Model training failed")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Training failed: {exc}"},
        )


@app.post("/predict", response_class=HTMLResponse)
async def handle_predict(request: Request, audio_file: UploadFile = File(...)):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "is_trained": False,
                "error": "The model has not been trained yet. Please train the model before predicting.",
            },
            status_code=503,
        )

    temp_file_path = None
    try:
        if not audio_file.filename:
            raise HTTPException(
                status_code=400, detail="No file was selected for upload."
            )

        filename = os.path.basename(audio_file.filename.replace("\\", "/"))
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in SUPPORTED_AUDIO_FORMATS:
            raise HTTPException(
                status_code=415,
                detail="Unsupported audio format. Upload a WAV, MP3, M4A, or FLAC file.",
            )

        with tempfile.NamedTemporaryFile(
            mode="wb", dir=UPLOADS_DIR, suffix=suffix, delete=False
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
                "transcript": transcript or "[No speech detected in audio file]",
                "prediction": prediction_result["prediction"],
                "confidence": prediction_result["confidence"],
                "detected_keywords": prediction_result["detected_keywords"],
            },
        )
    except HTTPException as exc:
        is_trained = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"is_trained": is_trained, "error": exc.detail},
            status_code=exc.status_code,
        )
    except Exception as exc:
        logger.exception("Audio analysis failed")
        is_trained = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "is_trained": is_trained,
                "error": f"Audio analysis failed: {exc}",
            },
            status_code=500,
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        await audio_file.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
