"""
import_real_dataset.py
Extracts and structures real-world garbage images from archive/garbage_classification
into dataset/train, dataset/val, and dataset/test with 70% / 15% / 15% split.
Target Classes:
  1. organic (from archive/garbage_classification/biological)
  2. plastic (from archive/garbage_classification/plastic)
  3. paper   (from archive/garbage_classification/paper and cardboard)
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive" / "garbage_classification"
DATASET_DIR = BASE_DIR / "dataset"

# Target sample counts per class (sufficient for high accuracy and fast CPU training)
MAX_PER_CLASS = 400

MAPPING = {
    "organic": ["biological"],
    "plastic": ["plastic"],
    "paper": ["paper", "cardboard"]
}

def is_valid_image(file_path: Path) -> bool:
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def main():
    print("=" * 60)
    print("IMPORTING REAL-WORLD GARBAGE DATASET")
    print("=" * 60)

    if not ARCHIVE_DIR.exists():
        print(f"Error: Archive directory not found at {ARCHIVE_DIR}")
        return

    # Create target directories
    for split in ["train", "val", "test"]:
        for cat in MAPPING.keys():
            (DATASET_DIR / split / cat).mkdir(parents=True, exist_ok=True)

    summary = {}

    for cat_name, source_subdirs in MAPPING.items():
        all_candidate_files = []
        for sdir in source_subdirs:
            folder = ARCHIVE_DIR / sdir
            if folder.exists():
                for f in folder.glob("*.*"):
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                        all_candidate_files.append(f)

        print(f"\nFound {len(all_candidate_files)} total images for '{cat_name}'.")

        # Shuffle candidates
        random.shuffle(all_candidate_files)

        # Validate images and collect up to MAX_PER_CLASS
        valid_files = []
        for f in all_candidate_files:
            if is_valid_image(f):
                valid_files.append(f)
            if len(valid_files) >= MAX_PER_CLASS:
                break

        print(f"Selected {len(valid_files)} verified real images for '{cat_name}'.")

        # Split 70% Train, 15% Val, 15% Test
        n = len(valid_files)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        n_test = n - n_train - n_val

        train_files = valid_files[:n_train]
        val_files = valid_files[n_train:n_train + n_val]
        test_files = valid_files[n_train + n_val:]

        summary[cat_name] = {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
            "total": n
        }

        # Copy to dataset structure
        for f in train_files:
            shutil.copy2(f, DATASET_DIR / "train" / cat_name / f.name)
        for f in val_files:
            shutil.copy2(f, DATASET_DIR / "val" / cat_name / f.name)
        for f in test_files:
            shutil.copy2(f, DATASET_DIR / "test" / cat_name / f.name)

    print("\n" + "=" * 60)
    print("DATASET IMPORT SUMMARY (REAL-WORLD IMAGES):")
    print("=" * 60)
    total_all = 0
    for cat_name, counts in summary.items():
        print(f"{cat_name.upper():<10} | Train: {counts['train']} | Val: {counts['val']} | Test: {counts['test']} | Total: {counts['total']}")
        total_all += counts['total']
    print("-" * 60)
    print(f"Grand Total Real Images: {total_all}")
    print("=" * 60)

if __name__ == "__main__":
    main()
