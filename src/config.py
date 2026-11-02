from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "waste_classifier.keras"
EVALUATION_DIR = BASE_DIR / "evaluation"
DATASET_DIR = BASE_DIR / "dataset"

# Classification Specification
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
CLASS_NAMES = ["Organic Waste", "Plastic Waste", "Paper Waste"]
NUM_CLASSES = len(CLASS_NAMES)

# File constraints
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
