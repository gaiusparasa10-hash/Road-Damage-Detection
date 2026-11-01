"""
=============================================================================
Project: AI-Based Road Damage Detection Using Deep Learning (CNN)
File: train_model.py
Description:
    Builds, compiles, and trains a beginner-friendly Convolutional Neural
    Network (CNN) using TensorFlow and Keras to classify road conditions
    into 'Normal' and 'Damaged'.
    
    Architecture:
    - Input: 128 x 128 x 3 RGB images
    - Layer 1: Rescaling (1./255) to normalize pixel values [0, 1]
    - Layer 2: Conv2D (32 filters, 3x3 kernel, ReLU activation)
    - Layer 3: MaxPooling2D (2x2 pool size)
    - Layer 4: Conv2D (64 filters, 3x3 kernel, ReLU activation)
    - Layer 5: MaxPooling2D (2x2 pool size)
    - Layer 6: Conv2D (128 filters, 3x3 kernel, ReLU activation)
    - Layer 7: MaxPooling2D (2x2 pool size)
    - Layer 8: Flatten (Flattens 2D feature maps to 1D vector)
    - Layer 9: Dense (128 neurons, ReLU activation)
    - Layer 10: Dropout (0.5 rate to prevent overfitting)
    - Output Layer: Dense (1 neuron, Sigmoid activation for binary classification)
=============================================================================
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------------
# HYPERPARAMETERS & CONFIGURATION
# ---------------------------------------------------------------------------
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
BATCH_SIZE = 32
EPOCHS = 10
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
NORMAL_DIR = os.path.join(PROCESSED_DIR, "Normal")
DAMAGED_DIR = os.path.join(PROCESSED_DIR, "Damaged")

MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "road_damage_model.keras")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
HISTORY_SAVE_PATH = os.path.join(RESULTS_DIR, "history.json")
CLASSES_SAVE_PATH = os.path.join(RESULTS_DIR, "class_names.json")


def verify_dataset_availability():
    """
    Verifies that processed images exist for both Normal and Damaged classes.
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    
    normal_count = 0
    damaged_count = 0
    
    if os.path.exists(NORMAL_DIR):
        normal_count = len([f for f in os.listdir(NORMAL_DIR) if f.lower().endswith(valid_exts)])
        
    if os.path.exists(DAMAGED_DIR):
        damaged_count = len([f for f in os.listdir(DAMAGED_DIR) if f.lower().endswith(valid_exts)])

    print("-" * 70)
    print(" DATASET AVAILABILITY CHECK")
    print("-" * 70)
    print(f"  Processed Normal Road Images  : {normal_count}")
    print(f"  Processed Damaged Road Images : {damaged_count}")
    print("-" * 70)

    if normal_count == 0 or damaged_count == 0:
        print("[ERROR] Insufficient dataset for training.")
        print("Both 'Normal' and 'Damaged' classes must contain at least a few images.")
        print("\nPlease run 'python preprocess.py' after placing raw images and XML annotations.")
        print("See the README.md for complete dataset setup instructions.")
        return False, normal_count, damaged_count

    return True, normal_count, damaged_count


def build_cnn_model():
    """
    Constructs the Sequential CNN model architecture with comprehensive comments
    explaining every single layer for beginner students.
    """
    print("\n[INFO] Building CNN Architecture...")
    
    model = models.Sequential([
        # -------------------------------------------------------------------
        # INPUT & RESCALING LAYER:
        # Accepts 128x128 RGB images (3 channels: Red, Green, Blue).
        # Normalizes raw pixel values from integers [0, 255] to floats [0.0, 1.0].
        # -------------------------------------------------------------------
        layers.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), name="input_layer"),
        layers.Rescaling(1.0 / 255, name="rescaling_layer"),

        # -------------------------------------------------------------------
        # CONVOLUTIONAL BLOCK 1:
        # Conv2D: 32 filters of size 3x3 to detect low-level edges and corners.
        # ReLU: Replaces negative values with 0, introducing non-linearity.
        # MaxPooling2D: Reduces spatial dimensions by 50% (from 128x128 to 64x64).
        # -------------------------------------------------------------------
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", name="conv2d_1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2d_1"),

        # -------------------------------------------------------------------
        # CONVOLUTIONAL BLOCK 2:
        # Conv2D: 64 filters of size 3x3 to detect mid-level texture patterns
        # (e.g., surface cracks, continuous fissures).
        # MaxPooling2D: Downsamples spatial dimensions (from 64x64 to 32x32).
        # -------------------------------------------------------------------
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", name="conv2d_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2d_2"),

        # -------------------------------------------------------------------
        # CONVOLUTIONAL BLOCK 3:
        # Conv2D: 128 filters of size 3x3 to detect high-level semantic shapes
        # (e.g., severe potholes, complex alligator cracks).
        # MaxPooling2D: Downsamples spatial dimensions (from 32x32 to 16x16).
        # -------------------------------------------------------------------
        layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", name="conv2d_3"),
        layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2d_3"),

        # -------------------------------------------------------------------
        # DENSE CLASSIFICATION HEAD:
        # Flatten: Converts the 3D feature maps (16 x 16 x 128) into a 1D vector (32,768).
        # Dense: 128 neurons fully connected to learn combinations of features.
        # Dropout: Randomly turns off 50% of neurons during training to prevent memorization/overfitting.
        # -------------------------------------------------------------------
        layers.Flatten(name="flatten_layer"),
        layers.Dense(units=128, activation="relu", name="dense_features"),
        layers.Dropout(rate=0.5, name="dropout_layer"),

        # -------------------------------------------------------------------
        # OUTPUT LAYER:
        # 1 neuron with Sigmoid activation function.
        # Outputs a probability value between 0.0 and 1.0.
        # -------------------------------------------------------------------
        layers.Dense(units=1, activation="sigmoid", name="output_prediction")
    ])

    # -----------------------------------------------------------------------
    # COMPILATION:
    # Optimizer: Adam (Adaptive Moment Estimation) adjusts learning rate dynamically.
    # Loss: Binary Crossentropy (standard loss function for two-class classification).
    # Metrics: Accuracy measures the proportion of correctly classified images.
    # -----------------------------------------------------------------------
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


def train_cnn():
    """
    Loads dataset, splits into 80% train / 20% validation, trains CNN for 10 epochs,
    and saves the trained model and training history.
    """
    print("=" * 70)
    print(" AI-BASED ROAD DAMAGE DETECTION - CNN MODEL TRAINING")
    print("=" * 70)

    # 1. Check if dataset exists
    is_ready, normal_count, damaged_count = verify_dataset_availability()
    if not is_ready:
        return

    # 2. Set random seeds for reproducible splitting
    tf.random.set_seed(RANDOM_SEED)

    # 3. Load datasets with 80% training / 20% validation split
    print(f"\n[INFO] Loading images from {PROCESSED_DIR}...")
    print(f"       Split ratio: 80% Training, 20% Validation (Seed: {RANDOM_SEED})")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR,
        validation_split=0.2,
        subset="training",
        seed=RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR,
        validation_split=0.2,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )

    class_names = train_dataset.class_names
    print(f"[INFO] Discovered Classes: {class_names}")
    # Label 0 corresponds to class_names[0], Label 1 corresponds to class_names[1]

    # Save class names mapping for predict.py & evaluate_model.py
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CLASSES_SAVE_PATH, "w") as f:
        json.dump(class_names, f, indent=4)

    # Optimize data loading pipeline for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # 4. Build Model
    model = build_cnn_model()

    # 5. Train Model
    print(f"\n[INFO] Starting model training for {EPOCHS} epochs with batch size {BATCH_SIZE}...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        verbose=1
    )

    # 6. Save Model
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\n[SUCCESS] Model successfully saved to: {MODEL_SAVE_PATH}")

    # 7. Save History (Accuracy and Loss values across epochs)
    history_dict = {
        "accuracy": [float(val) for val in history.history.get("accuracy", [])],
        "val_accuracy": [float(val) for val in history.history.get("val_accuracy", [])],
        "loss": [float(val) for val in history.history.get("loss", [])],
        "val_loss": [float(val) for val in history.history.get("val_loss", [])]
    }
    with open(HISTORY_SAVE_PATH, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"[SUCCESS] Training history saved to: {HISTORY_SAVE_PATH}")

    # Print final epoch summary
    final_train_acc = history_dict["accuracy"][-1] * 100 if history_dict["accuracy"] else 0
    final_val_acc = history_dict["val_accuracy"][-1] * 100 if history_dict["val_accuracy"] else 0
    print("-" * 70)
    print(f" Final Training Accuracy   : {final_train_acc:.2f}%")
    print(f" Final Validation Accuracy : {final_val_acc:.2f}%")
    print("-" * 70)
    print("Next step: Run 'python evaluate_model.py' to generate evaluation graphs and metrics.")
    print("=" * 70)


if __name__ == "__main__":
    train_cnn()
