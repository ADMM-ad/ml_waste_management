import io
import os
from pathlib import Path
from PIL import Image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from src.config import CLASS_NAMES
from src.inference import predict_waste, is_model_loaded, get_model

def test_direct_inference():
    print("1. Checking is_model_loaded()...")
    assert is_model_loaded() is True
    print(" -> Model exists on disk.")

    print("2. Checking model loading...")
    model = get_model()
    assert model is not None
    print(f" -> Model loaded: {type(model).__name__}")

    print("3. Testing inference on sample image...")
    img = Image.new("RGB", (224, 224), color=(60, 160, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    res = predict_waste(buf.getvalue())
    print(f" -> Result: {res}")
    assert res["predicted_class"] in CLASS_NAMES
    assert 0.0 <= res["confidence_score"] <= 1.0

    print("ALL UNIT & INFERENCE TESTS PASSED!")

if __name__ == "__main__":
    test_direct_inference()
