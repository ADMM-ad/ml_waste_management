import io
import time
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

from src.config import MODEL_PATH, CLASS_NAMES, IMAGE_SIZE

_model_instance = None

def get_model():
    """Singleton getter for the loaded TensorFlow / Keras model."""
    global _model_instance
    if _model_instance is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. Run train.py first to generate the model."
            )
        import tensorflow as tf
        _model_instance = tf.keras.models.load_model(str(MODEL_PATH))
    return _model_instance

def is_model_loaded() -> bool:
    """Check if model file exists and is ready."""
    return MODEL_PATH.exists()

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Validate, open, and resize to (224, 224) RGB with float32 [0, 255] values.
    The Keras model pipeline already includes tf.keras.applications.mobilenet_v2.preprocess_input.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Invalid or corrupted image format: {exc}") from exc

    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_waste(image_bytes: bytes) -> Dict[str, Any]:
    """
    Executes end-to-end inference on the input image bytes.
    Returns:
        dict containing predicted_class, confidence_score, probabilities, inference_time_ms.
    """
    model = get_model()
    start_time = time.perf_counter()

    img_array = preprocess_image(image_bytes)
    raw_preds = model.predict(img_array, verbose=0)
    probabilities_list = raw_preds[0]

    max_idx = int(np.argmax(probabilities_list))
    predicted_class = CLASS_NAMES[max_idx]
    confidence_score = float(probabilities_list[max_idx])

    probabilities_dict = {
        CLASS_NAMES[i]: float(round(probabilities_list[i], 4))
        for i in range(len(CLASS_NAMES))
    }

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "predicted_class": predicted_class,
        "confidence_score": round(confidence_score, 4),
        "probabilities": probabilities_dict,
        "inference_time_ms": round(elapsed_ms, 2)
    }
