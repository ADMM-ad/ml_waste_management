"""
train.py
End-to-end training and evaluation script for MobileNetV2 Waste Classification
trained on real-world garbage dataset (Organic, Plastic, Paper).
Outputs:
  - evaluation/training_history.png
  - evaluation/confusion_matrix.png
  - evaluation/classification_report.txt
  - models/waste_classifier.keras
"""
import os
import sys
from pathlib import Path

# Ensure matplotlib uses a local writable cache directory
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent / ".mpl_cache")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    DATASET_DIR,
    MODELS_DIR,
    MODEL_PATH,
    EVALUATION_DIR,
    IMAGE_SIZE,
    CLASS_NAMES,
    NUM_CLASSES,
)

def build_model():
    """Builds transfer learning model using MobileNetV2."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")

    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = data_augmentation(inputs)
    # MobileNetV2 preprocessing maps [0, 255] -> [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="WasteClassifierMobileNetV2")
    return model, base_model

def main():
    print("=" * 60)
    print("STARTING WASTE CLASSIFIER TRAINING ON REAL-WORLD DATASET")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    test_dir = DATASET_DIR / "test"

    if not train_dir.exists():
        print(f"Dataset directory not found at {train_dir}. Run import_real_dataset.py first!")
        sys.exit(1)

    batch_size = 32

    print("\n1. Loading datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=['organic', 'plastic', 'paper'],
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=['organic', 'plastic', 'paper'],
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=['organic', 'plastic', 'paper'],
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False
    )

    # Performance optimization
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    print("\n2. Building and compiling MobileNetV2 Transfer Learning model...")
    model, base_model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    epochs = 12
    print(f"\n3. Training model for up to {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    print(f"\n4. Saving trained model artifact to {MODEL_PATH}...")
    model.save(MODEL_PATH)

    print("\n5. Evaluating on independent Test Set (180 Real Photos)...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"--> Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%")

    # Generate predictions for classification report and confusion matrix
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.argmax(np.array(y_pred_probs), axis=1)

    # Classification Report
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
    report_header = f"WASTE CLASSIFICATION REPORT (MobileNetV2 on Real-World Photos)\nTest Accuracy: {test_acc * 100:.2f}%\n\n"
    full_report = report_header + report
    print("\n" + full_report)

    report_path = EVALUATION_DIR / "classification_report.txt"
    report_path.write_text(full_report, encoding="utf-8")
    print(f"Saved classification report to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title(f'Confusion Matrix (Test Acc: {test_acc * 100:.2f}%)', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.tight_layout()
    cm_path = EVALUATION_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

    # Training History Plot
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy', color='#16a34a', linewidth=2)
    plt.plot(val_acc, label='Validation Accuracy', color='#2563eb', linewidth=2)
    plt.title('Training & Validation Accuracy', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss', color='#dc2626', linewidth=2)
    plt.plot(val_loss, label='Validation Loss', color='#f59e0b', linewidth=2)
    plt.title('Training & Validation Loss', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_path = EVALUATION_DIR / "training_history.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"Saved training history plot to {hist_path}")

    print("\n" + "=" * 60)
    print("ALL TRAINING & EVALUATION TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
