"""
test_api.py
Automated test suite for FastAPI Waste Classifier endpoints.
"""
import io
import os
from pathlib import Path
from PIL import Image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi.testclient import TestClient
from src.main import app
from src.config import CLASS_NAMES

def create_dummy_image(color=(34, 139, 34)) -> bytes:
    img = Image.new("RGB", (224, 224), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def run_tests():
    with TestClient(app) as client:
        print("Testing /health...")
        res = client.get("/health")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["classes"] == CLASS_NAMES
        print(" -> /health OK!")

        print("Testing /predict (valid image)...")
        img_bytes = create_dummy_image((50, 160, 40))
        files = {"file": ("sample.jpg", img_bytes, "image/jpeg")}
        res = client.post("/predict", files=files)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        pred = res.json()
        assert pred["success"] is True
        assert pred["data"]["predicted_class"] in CLASS_NAMES
        print(f" -> /predict OK! Predicted: {pred['data']['predicted_class']} (Confidence: {pred['data']['confidence_score']})")

        print("Testing /predict (unsupported media type)...")
        files = {"file": ("note.txt", b"hello", "text/plain")}
        res = client.post("/predict", files=files)
        assert res.status_code == 415
        print(" -> /predict unsupported media OK!")

        print("Testing /predict (empty file)...")
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        res = client.post("/predict", files=files)
        assert res.status_code == 422
        print(" -> /predict empty file OK!")

    print("\nALL FASTAPI INTEGRATION TESTS PASSED 100%!")

if __name__ == "__main__":
    run_tests()
