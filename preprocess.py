"""
=============================================================================
Project: AI-Based Road Damage Detection Using Deep Learning (CNN)
File: preprocess.py
Description:
    Preprocesses the raw Road Damage Dataset (RDD2022) into two binary classes:
    - Normal: Roads with zero damage annotations or verified undamaged surfaces.
    - Damaged: Roads with one or more damage annotations (Cracks, Potholes, etc.).
    
    Steps performed:
    1. Checks raw images and corresponding Pascal VOC XML annotation files.
    2. Filters out images with missing or corrupted annotations.
    3. Resizes images to standard CNN input dimension (128 x 128).
    4. Converts images to 3-channel RGB format.
    5. Saves structured outputs into data/processed/Normal/ and data/processed/Damaged/.
=============================================================================
"""

import os
import sys
import xml.etree.ElementTree as ET
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIGURATION & PATH DEFINITIONS
# ---------------------------------------------------------------------------
# Target image resolution for our beginner-friendly CNN model
TARGET_IMAGE_SIZE = (128, 128)

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
RAW_ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "raw", "annotations")
OPTIONAL_NORMAL_DIR = os.path.join(BASE_DIR, "data", "raw", "normal_images")

PROCESSED_NORMAL_DIR = os.path.join(BASE_DIR, "data", "processed", "Normal")
PROCESSED_DAMAGED_DIR = os.path.join(BASE_DIR, "data", "processed", "Damaged")

# Supported image file extensions
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Known RDD2022 damage labels:
# D00: Longitudinal Crack (wheel mark)
# D10: Longitudinal Crack (construction joint)
# D20: Alligator Crack
# D40: Pothole / rutting / bump / separation
KNOWN_DAMAGE_LABELS = {"D00", "D10", "D20", "D40", "D43", "D44"}


def ensure_directories():
    """Ensure that all destination directories exist."""
    os.makedirs(PROCESSED_NORMAL_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DAMAGED_DIR, exist_ok=True)


def parse_xml_annotation(xml_path):
    """
    Parses a Pascal VOC format XML file to count damage objects.
    
    Returns:
        int: Number of damage objects found. Returns -1 if the XML is invalid.
    """
    if not os.path.exists(xml_path):
        return -1  # Explicit indicator that annotation is missing

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        objects = root.findall("object")
        damage_count = 0
        
        for obj in objects:
            name_elem = obj.find("name")
            if name_elem is not None and name_elem.text:
                label = name_elem.text.strip()
                # In RDD2022, damage labels start with 'D' (e.g., D00, D10, D20, D40)
                # Any recognized damage or non-empty object represents damage.
                if label in KNOWN_DAMAGE_LABELS or label.startswith("D") or "damage" in label.lower() or "crack" in label.lower() or "pothole" in label.lower():
                    damage_count += 1
                else:
                    # In case of custom general object annotations
                    damage_count += 1

        return damage_count
    except Exception as error:
        print(f"[WARNING] Could not parse XML: {xml_path} (Error: {error})")
        return -1


def process_single_image(image_path, destination_folder, filename):
    """
    Opens an image, converts it to RGB, resizes to (128, 128), and saves it.
    
    Returns:
        bool: True if successful, False if image was corrupt or unreadable.
    """
    try:
        with Image.open(image_path) as img:
            # Convert grayscale or RGBA to 3-channel RGB
            rgb_img = img.convert("RGB")
            # Resize image with high-quality resampling filter
            resized_img = rgb_img.resize(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            # Save processed image as JPEG
            output_filename = os.path.splitext(filename)[0] + ".jpg"
            save_path = os.path.join(destination_folder, output_filename)
            resized_img.save(save_path, "JPEG", quality=90)
            return True
    except Exception as error:
        print(f"[WARNING] Skipping unreadable image: {image_path} (Error: {error})")
        return False


def run_preprocessing():
    """
    Main function to orchestrate dataset preprocessing.
    """
    print("=" * 70)
    print(" AI-BASED ROAD DAMAGE DETECTION - DATASET PREPROCESSING")
    print("=" * 70)
    
    ensure_directories()
    
    # 1. Check if raw images directory exists and contains images
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"[ERROR] Raw images folder not found at: {RAW_IMAGES_DIR}")
        print_dataset_guide()
        return

    all_raw_files = [f for f in os.listdir(RAW_IMAGES_DIR) if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
    
    # Also check if user provided dedicated normal images
    dedicated_normal_files = []
    if os.path.exists(OPTIONAL_NORMAL_DIR):
        dedicated_normal_files = [f for f in os.listdir(OPTIONAL_NORMAL_DIR) if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]

    if not all_raw_files and not dedicated_normal_files:
        print("[INFO] No raw road images found in data/raw/images/.")
        print_dataset_guide()
        return

    print(f"[INFO] Found {len(all_raw_files)} raw image(s) in {RAW_IMAGES_DIR}")
    if dedicated_normal_files:
        print(f"[INFO] Found {len(dedicated_normal_files)} verified normal image(s) in {OPTIONAL_NORMAL_DIR}")

    count_damaged = 0
    count_normal = 0
    count_skipped_missing_xml = 0
    count_skipped_corrupt = 0

    # 2. Process RDD2022 images with XML annotations
    for filename in all_raw_files:
        image_path = os.path.join(RAW_IMAGES_DIR, filename)
        base_name, _ = os.path.splitext(filename)
        xml_filename = f"{base_name}.xml"
        xml_path = os.path.join(RAW_ANNOTATIONS_DIR, xml_filename)

        # Parse XML annotation
        damage_count = parse_xml_annotation(xml_path)

        if damage_count == -1:
            # Annotation is missing or corrupted -> SKIP (Do NOT treat as normal!)
            count_skipped_missing_xml += 1
            continue

        if damage_count > 0:
            # Damage detected in annotation -> Save to Damaged/
            success = process_single_image(image_path, PROCESSED_DAMAGED_DIR, filename)
            if success:
                count_damaged += 1
            else:
                count_corrupt += 1
        elif damage_count == 0:
            # Explicit annotation verifying 0 damage objects -> Save to Normal/
            success = process_single_image(image_path, PROCESSED_NORMAL_DIR, filename)
            if success:
                count_normal += 1
            else:
                count_corrupt += 1

    # 3. Process optional dedicated normal road images (if provided)
    for filename in dedicated_normal_files:
        image_path = os.path.join(OPTIONAL_NORMAL_DIR, filename)
        success = process_single_image(image_path, PROCESSED_NORMAL_DIR, f"norm_{filename}")
        if success:
            count_normal += 1
        else:
            count_skipped_corrupt += 1

    # 4. Display Preprocessing Summary
    print("-" * 70)
    print(" PREPROCESSING SUMMARY")
    print("-" * 70)
    print(f"  Processed Normal Road Images  : {count_normal}")
    print(f"  Processed Damaged Road Images : {count_damaged}")
    print(f"  Skipped (Missing/Bad XML)     : {count_skipped_missing_xml}")
    print(f"  Skipped (Corrupt Images)      : {count_skipped_corrupt}")
    print(f"  Total Clean Processed Images  : {count_normal + count_damaged}")
    print("-" * 70)

    if count_normal == 0 and count_damaged > 0:
        print("[NOTE] RDD2022 focuses primarily on damaged road instances.")
        print("       To achieve balanced binary classification, please provide verified")
        print("       undamaged road images inside 'data/raw/normal_images/' or supply")
        print("       XML annotations with 0 objects, then rerun preprocess.py.")
    elif count_normal > 0 and count_damaged > 0:
        print("[SUCCESS] Dataset preprocessing completed! You can now run 'python train_model.py'.")
    print("=" * 70)


def print_dataset_guide():
    """Prints a beginner-friendly guide for setting up the RDD2022 dataset."""
    print("""
=============================================================================
                      DATASET SETUP INSTRUCTIONS
=============================================================================
1. DOWNLOAD THE RDD2022 DATASET:
   - Official Paper & Repo: https://github.com/sekilab/RoadDamageDetector
   - Download link: https://doi.org/10.6084/m9.figshare.21431547.v1
   - Choose one country subset to start (e.g., Japan, India, or United_States).

2. PLACE RAW FILES INTO THE PROJECT:
   - Extract road images (.jpg) into:
     Road-Damage-Detection/data/raw/images/
   - Extract matching Pascal VOC XML annotations (.xml) into:
     Road-Damage-Detection/data/raw/annotations/

3. NORMAL ROAD IMAGES (UNDAMAGED ROADS):
   - Because RDD2022 focuses primarily on damaged road examples, you can also
     place clear, undamaged asphalt/concrete road images into:
     Road-Damage-Detection/data/raw/normal_images/
   - The script will automatically resize, convert to RGB, and include them in
     data/processed/Normal/.

4. RUN PREPROCESSING:
   python preprocess.py
=============================================================================
""")


if __name__ == "__main__":
    run_preprocessing()
