import os
import shutil
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from train import train_model, MODEL_PATH, VECTORIZER_PATH
from predict import predict_call
from whisper_service import transcribe_audio

app = FastAPI(title="Callytix - AI-Based Fraud Phone Call Detection System")

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_home(request: Request):
    """
    Renders the homepage (index.html).
    Passes information on whether the model is already trained.
    """
    is_trained = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "is_trained": is_trained}
    )

@app.post("/train")
async def handle_train():
    """
    Trains the TF-IDF Vectorizer and Multinomial Naive Bayes model
    on the Kaggle dataset. Returns the evaluation metrics as JSON.
    """
    try:
        metrics = train_model()
        return JSONResponse(content={
            "success": True,
            "message": "Model trained successfully!",
            "metrics": metrics
        })
    except Exception as e:
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
            "index.html",
            {
                "request": request,
                "is_trained": False,
                "error": "The model has not been trained yet. Please train the model before predicting."
            }
        )
        
    # 2. Validate file uploaded
    if not audio_file.filename:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "is_trained": True,
                "error": "No file was selected for upload."
            }
        )

    # Save uploaded file temporarily
    temp_file_path = os.path.join("uploads", audio_file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
            
        # 3. Transcribe audio using Whisper
        try:
            transcript = transcribe_audio(temp_file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Speech-to-text transcription failed: {str(e)}")
            
        # 4. Predict using Naive Bayes model
        try:
            prediction_result = predict_call(transcript)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
            
        # 5. Render results page
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "filename": audio_file.filename,
                "transcript": transcript if transcript else "[No speech detected in audio file]",
                "prediction": prediction_result["prediction"],
                "confidence": prediction_result["confidence"],
                "detected_keywords": prediction_result["detected_keywords"]
            }
        )
        
    except Exception as e:
        # Render homepage with error message on exception
        is_trained = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "is_trained": is_trained,
                "error": f"An error occurred: {str(e)}"
            }
        )
    finally:
        # Clean up the uploaded audio file to save disk space
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
