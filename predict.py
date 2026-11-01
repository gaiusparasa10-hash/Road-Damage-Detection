"""
=============================================================================
Project: AI-Based Road Damage Detection Using Deep Learning (CNN)
File: predict.py
Description:
    Stand-alone command-line inference script. Takes a road image file,
    preprocesses it to 128x128 RGB, loads the trained CNN model, and outputs:
    - Predicted Road Condition: NORMAL ROAD or DAMAGED ROAD
    - Confidence Score (%)
=============================================================================
"""

import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------------------------------------------------------------------
# PATH DEFINITIONS & CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "road_damage_model.keras")
IMAGE_SIZE = (128, 128)


def load_cnn_model():
    """
    Loads and returns the trained Keras CNN model.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        print("Please train the model first by running: python train_model.py")
        sys.exit(1)
        
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as error:
        print(f"[ERROR] Failed to load trained model: {error}")
        sys.exit(1)


def preprocess_image(image_path):
    """
    Opens an image, converts to 3-channel RGB, resizes to (128, 128),
    and expands dimensions to create a batch of 1: (1, 128, 128, 3).
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file does not exist at: {image_path}")
        return None

    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            resized_img = rgb_img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            img_array = np.array(resized_img, dtype=np.float32)
            # Add batch dimension: shape becomes (1, 128, 128, 3)
            # Note: Model includes a Rescaling(1./255) layer at input.
            batch_input = np.expand_dims(img_array, axis=0)
            return batch_input
    except Exception as error:
        print(f"[ERROR] Could not read/process image: {error}")
        return None


def predict_road_condition(image_path, model=None):
    """
    Predicts whether a given road image is 'NORMAL ROAD' or 'DAMAGED ROAD'.
    
    Returns:
        tuple: (predicted_label_str, confidence_percentage_float)
    """
    if model is None:
        model = load_cnn_model()

    input_data = preprocess_image(image_path)
    if input_data is None:
        return None, None

    # Predict using the model
    # Sigmoid output represents P(Normal) because alphabetically 'Damaged' is index 0 and 'Normal' is index 1.
    prediction = model.predict(input_data, verbose=0)
    normal_probability = float(prediction[0][0])

    if normal_probability >= 0.5:
        predicted_condition = "NORMAL ROAD"
        confidence = normal_probability * 100.0
    else:
        predicted_condition = "DAMAGED ROAD"
        confidence = (1.0 - normal_probability) * 100.0

    return predicted_condition, confidence


def main():
    print("=" * 70)
    print(" AI-BASED ROAD DAMAGE DETECTION - INFERENCE")
    print("=" * 70)

    # Check command-line argument for image path
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
    else:
        print("Usage: python predict.py <path_to_image>")
        target_image = input("Enter path to road image: ").strip().strip('"').strip("'")

    if not target_image:
        print("[ERROR] No image path provided.")
        return

    condition, confidence = predict_road_condition(target_image)
    if condition is not None:
        print("-" * 70)
        print(f" Image File  : {target_image}")
        print(f" Prediction  : {condition}")
        print(f" Confidence  : {confidence:.2f}%")
        print("-" * 70)
        print("=" * 70)


if __name__ == "__main__":
    main()
