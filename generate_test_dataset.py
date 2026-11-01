"""
=============================================================================
Project: AI-Based Road Damage Detection Using Deep Learning (CNN)
File: generate_test_dataset.py
Description:
    Generates a lightweight sample dataset with realistic asphalt road patterns
    and Pascal VOC XML annotations.
    
    This script is specifically provided so students and evaluators can:
    1. Test the full preprocessing pipeline (preprocess.py)
    2. Train the CNN model (train_model.py)
    3. Evaluate performance and generate plots (evaluate_model.py)
    4. Run real-time predictions via Streamlit (app.py)
    ...even before downloading the large multi-gigabyte RDD2022 dataset!
=============================================================================
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
RAW_ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "raw", "annotations")

# Ensure target directories exist
os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
os.makedirs(RAW_ANNOTATIONS_DIR, exist_ok=True)


def create_road_texture(width=256, height=256):
    """Creates a realistic gray asphalt/concrete roadway texture."""
    base_gray = random.randint(70, 95)
    noise = np.random.normal(loc=base_gray, scale=12, size=(height, width, 3))
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(noise, mode="RGB")
    return img


def draw_damage_on_road(img, damage_type="crack"):
    """Draws synthetic crack or pothole distress features on road texture."""
    draw = ImageDraw.Draw(img)
    width, height = img.size

    if damage_type == "pothole":
        # Draw dark pothole depression with irregular border
        cx, cy = random.randint(60, width - 60), random.randint(60, height - 60)
        rx, ry = random.randint(25, 45), random.randint(20, 35)
        # Deep shadow inside pothole
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(25, 25, 25), outline=(15, 15, 15), width=3)
        # Inner depth contour
        draw.ellipse([cx - rx // 2, cy - ry // 2, cx + rx // 2, cy + ry // 2], fill=(10, 10, 10))
        bbox = (cx - rx, cy - ry, cx + rx, cy + ry)
        label = "D40"  # RDD2022 label for Pothole/Rutting

    elif damage_type == "alligator":
        # Interconnected polygonal cracks
        start_x, start_y = random.randint(40, width - 80), random.randint(40, height - 80)
        for _ in range(6):
            nx = start_x + random.randint(-30, 30)
            ny = start_y + random.randint(-30, 30)
            draw.line([(start_x, start_y), (nx, ny)], fill=(20, 20, 20), width=2)
            start_x, start_y = max(10, min(width - 10, nx)), max(10, min(height - 10, ny))
        bbox = (30, 30, width - 30, height - 30)
        label = "D20"  # RDD2022 label for Alligator crack

    else:
        # Longitudinal or transverse crack line
        curr_x = random.randint(50, width - 50)
        curr_y = random.randint(20, 50)
        points = [(curr_x, curr_y)]
        for _ in range(7):
            curr_x += random.randint(-12, 12)
            curr_y += random.randint(20, 35)
            curr_x = max(10, min(width - 10, curr_x))
            curr_y = max(10, min(height - 10, curr_y))
            points.append((curr_x, curr_y))
        draw.line(points, fill=(20, 20, 20), width=random.randint(2, 4))
        bbox = (min(p[0] for p in points), min(p[1] for p in points),
                max(p[0] for p in points), max(p[1] for p in points))
        label = "D00"  # RDD2022 label for Longitudinal crack

    return img, label, bbox


def write_pascal_voc_xml(xml_path, filename, width, height, objects):
    """Generates standard Pascal VOC XML annotation file."""
    obj_xml_lines = []
    for label, (xmin, ymin, xmax, ymax) in objects:
        obj_xml_lines.append(f"""  <object>
    <name>{label}</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>{int(max(0, xmin))}</xmin>
      <ymin>{int(max(0, ymin))}</ymin>
      <xmax>{int(min(width, xmax))}</xmax>
      <ymax>{int(min(height, ymax))}</ymax>
    </bndbox>
  </object>""")

    objects_xml = "\n".join(obj_xml_lines)
    xml_content = f"""<annotation>
  <folder>images</folder>
  <filename>{filename}</filename>
  <size>
    <width>{width}</width>
    <height>{height}</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
{objects_xml}
</annotation>"""

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content.strip())


def generate_sample_dataset(num_damaged=25, num_normal=25):
    """Generates balanced sample road dataset for quick local verification."""
    print("=" * 70)
    print(" GENERATING SAMPLE TEST ROAD DATASET FOR VERIFICATION")
    print("=" * 70)
    
    random.seed(42)
    np.random.seed(42)
    
    # 1. Generate Damaged Road Images & XMLs
    damage_types = ["longitudinal", "alligator", "pothole"]
    for i in range(1, num_damaged + 1):
        filename = f"sample_damaged_{i:03d}.jpg"
        img = create_road_texture()
        dtype = damage_types[(i - 1) % len(damage_types)]
        img, label, bbox = draw_damage_on_road(img, damage_type=dtype)

        # Save Image
        img_path = os.path.join(RAW_IMAGES_DIR, filename)
        img.save(img_path, "JPEG", quality=95)

        # Save XML Annotation with damage object
        xml_filename = f"sample_damaged_{i:03d}.xml"
        xml_path = os.path.join(RAW_ANNOTATIONS_DIR, xml_filename)
        write_pascal_voc_xml(xml_path, filename, 256, 256, [(label, bbox)])

    # 2. Generate Normal Road Images & XMLs (0 damage objects)
    for i in range(1, num_normal + 1):
        filename = f"sample_normal_{i:03d}.jpg"
        img = create_road_texture()
        # Add smooth road markings or lane paint (undamaged road feature)
        if i % 2 == 0:
            draw = ImageDraw.Draw(img)
            draw.rectangle([120, 0, 136, 256], fill=(210, 210, 210))  # White lane stripe

        # Save Image
        img_path = os.path.join(RAW_IMAGES_DIR, filename)
        img.save(img_path, "JPEG", quality=95)

        # Save XML Annotation with 0 objects (explicit normal road)
        xml_filename = f"sample_normal_{i:03d}.xml"
        xml_path = os.path.join(RAW_ANNOTATIONS_DIR, xml_filename)
        write_pascal_voc_xml(xml_path, filename, 256, 256, [])

    print(f"[SUCCESS] Generated {num_damaged} Damaged road images with XMLs.")
    print(f"[SUCCESS] Generated {num_normal} Normal road images with XMLs.")
    print(f"Files saved in:")
    print(f"  Images      : {RAW_IMAGES_DIR}")
    print(f"  Annotations : {RAW_ANNOTATIONS_DIR}")
    print("\nYou can now run:")
    print("  1. python preprocess.py")
    print("  2. python train_model.py")
    print("  3. python evaluate_model.py")
    print("  4. streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    generate_sample_dataset()
