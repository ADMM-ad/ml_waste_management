"""
train.py
End-to-end training and evaluation script for MobileNetV2 Waste Classification.
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
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")

    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = data_augmentation(inputs)
    # MobileNetV2 preprocessing maps [0, 255] -> [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="WasteClassifierMobileNetV2")
    return model

def main():
    print("=" * 60)
    print("STARTING WASTE CLASSIFIER TRAINING & EVALUATION PIPELINE")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    test_dir = DATASET_DIR / "test"

    if not train_dir.exists():
        print(f"Dataset directory not found at {train_dir}. Run prepare_dataset.py first!")
        sys.exit(1)

    batch_size = 16

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

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    print("\n2. Building MobileNetV2 Transfer Learning Model...")
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("\n3. Training model...")
    epochs = 20
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    print("\n4. Exporting training history curves...")
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='#2563eb', lw=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='#dc2626', lw=2)
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Crossentropy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='#2563eb', lw=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#16a34a', lw=2)
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    history_plot_path = EVALUATION_DIR / "training_history.png"
    plt.savefig(history_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {history_plot_path}")

    print("\n5. Evaluating model on test set...")
    # Collect predictions and true labels
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    test_accuracy = float(np.mean(y_true == y_pred))
    print(f"\n==========================================")
    print(f"TEST ACCURACY: {test_accuracy * 100:.2f}%")
    print(f"==========================================")

    # Export Classification Report
    target_names = ["Organic Waste", "Plastic Waste", "Paper Waste"]
    report_text = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    report_path = EVALUATION_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("WASTE CLASSIFICATION REPORT (MobileNetV2)\n")
        f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n")
        f.write(report_text)
    print(f"Saved: {report_path}")
    print("\n" + report_text)

    # Export Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True
    )
    plt.title(f'Confusion Matrix (Test Accuracy: {test_accuracy * 100:.1f}%)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_plot_path = EVALUATION_DIR / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_plot_path}")

    # Check target criteria >= 80%
    if test_accuracy >= 0.80:
        print("\nTarget passing criterion achieved (>= 80.0%)! Saving final model...")
        model.save(str(MODEL_PATH))
        print(f"Model saved successfully to: {MODEL_PATH}")
    else:
        print(f"\nWARNING: Test accuracy ({test_accuracy*100:.2f}%) did not reach 80.0% target.")
        model.save(str(MODEL_PATH))

if __name__ == "__main__":
    main()
