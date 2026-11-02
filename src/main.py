from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import CLASS_NAMES, MAX_FILE_SIZE_BYTES, ALLOWED_MIME_TYPES
from src.inference import get_model, is_model_loaded, predict_waste

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WasteClassifierAPI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload model if available
    logger.info("Initializing Waste Classifier Service...")
    if is_model_loaded():
        try:
            get_model()
            logger.info("MobileNetV2 Waste Classifier model loaded successfully into memory.")
        except Exception as e:
            logger.warning(f"Failed to preload model at startup: {e}")
    else:
        logger.warning("Model file not yet found on disk. Run train.py to train and create it.")
    yield
    logger.info("Shutting down Waste Classifier Service.")

app = FastAPI(
    title="Waste Classification Inference Microservice",
    description="REST API for classifying waste into Organic, Plastic, and Paper using MobileNetV2",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", summary="Healthcheck and model status")
async def health():
    return {
        "status": "healthy",
        "model_loaded": is_model_loaded(),
        "classes": CLASS_NAMES
    }

@app.post("/predict", summary="Classify waste image")
async def predict(file: UploadFile = File(...)):
    # 1. Validate content type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_MIME_TYPES:
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={
                "success": false,
                "message": f"Unsupported media type '{content_type}'. Allowed: {ALLOWED_MIME_TYPES}",
                "data": None,
                "errors": ["unsupported_media_type"]
            }
        )

    # 2. Read bytes and validate file size
    try:
        image_bytes = await file.read()
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": false,
                "message": f"Failed to read uploaded file: {str(e)}",
                "data": None,
                "errors": ["read_error"]
            }
        )

    if len(image_bytes) == 0:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": false,
                "message": "Uploaded file is empty.",
                "data": None,
                "errors": ["empty_file"]
            }
        )

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "success": false,
                "message": f"File size exceeds maximum allowable limit of 5MB.",
                "data": None,
                "errors": ["file_too_large"]
            }
        )

    # 3. Model execution
    try:
        prediction_result = predict_waste(image_bytes)
        return {
            "success": True,
            "message": "Image classified successfully",
            "data": prediction_result
        }
    except ValueError as ve:
        # Image corruption or format decoding failure
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "message": str(ve),
                "data": None,
                "errors": ["corrupted_or_invalid_image"]
            }
        )
    except FileNotFoundError as fnf:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "success": False,
                "message": str(fnf),
                "data": None,
                "errors": ["model_not_trained"]
            }
        )
    except Exception as exc:
        logger.exception("Unexpected error during inference: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": f"Inference execution error: {str(exc)}",
                "data": None,
                "errors": ["internal_server_error"]
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
