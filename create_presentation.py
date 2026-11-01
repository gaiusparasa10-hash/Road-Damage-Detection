"""
=============================================================================
Project: AI-Based Road Damage Detection Using Deep Learning (CNN)
File: create_presentation.py
Description:
    Generates a professional 10-slide PowerPoint presentation (.pptx)
    following the executive high-contrast dark theme reference.
=============================================================================
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------------------------------------------------------------------------
# COLOR PALETTE DEFINITIONS (Modern Executive Dark Theme)
# ---------------------------------------------------------------------------
BG_DARK = RGBColor(11, 15, 25)         # Deep Navy / Midnight (#0B0F19)
SURFACE_CARD = RGBColor(22, 32, 51)    # Card surface slate (#162033)
CARD_BORDER = RGBColor(45, 60, 85)     # Subtle card border (#2D3C55)
PRIMARY_BLUE = RGBColor(59, 130, 246)  # Electric Blue (#3B82F6)
CYAN_ACCENT = RGBColor(6, 182, 212)    # Tech Cyan (#06B6D4)
PURPLE_ACCENT = RGBColor(139, 92, 246) # Violet (#8B5CF6)
TEXT_WHITE = RGBColor(248, 250, 252)   # Pure White (#F8FAFC)
TEXT_MUTED = RGBColor(148, 163, 184)   # Slate Muted (#94A3B8)
TEXT_ACCENT = RGBColor(96, 165, 250)   # Light Blue (#60A5FA)
GREEN_ACCENT = RGBColor(16, 185, 129)  # Emerald (#10B981)
RED_ACCENT = RGBColor(239, 68, 68)     # Crimson / Coral (#EF4444)
AMBER_ACCENT = RGBColor(245, 158, 11)  # Amber (#F59E0B)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PPTX = os.path.join(BASE_DIR, "Road_Damage_Detection_Presentation.pptx")
PREVIEW_IMG = os.path.join(BASE_DIR, "screenshots", "app_preview.png")
CM_IMG = os.path.join(BASE_DIR, "results", "confusion_matrix.png")
ACC_IMG = os.path.join(BASE_DIR, "results", "accuracy_plot.png")


def init_presentation():
    """Initializes 16:9 widescreen presentation."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs


def add_dark_background(slide, width_in=13.333, height_in=7.5):
    """Draws solid dark backdrop for slide."""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(width_in), Inches(height_in))
    shape.fill.solid()
    shape.fill.fore_color.rgb = BG_DARK
    shape.line.color.rgb = BG_DARK
    return shape


def add_header(slide, eyebrow_text, main_title, subtitle_text=None):
    """Creates a standardized modern header block."""
    # Eyebrow (Category Tracker)
    eyebrow_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.45), Inches(11.7), Inches(0.4))
    tf_eye = eyebrow_box.text_frame
    tf_eye.word_wrap = True
    p_eye = tf_eye.paragraphs[0]
    p_eye.text = eyebrow_text.upper()
    p_eye.font.size = Pt(11)
    p_eye.font.bold = True
    p_eye.font.color.rgb = PRIMARY_BLUE

    # Main Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.8))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = main_title
    p_title.font.size = Pt(26)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE

    # Subtitle (Optional)
    if subtitle_text:
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.7), Inches(0.4))
        tf_sub = sub_box.text_frame
        tf_sub.word_wrap = True
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = subtitle_text
        p_sub.font.size = Pt(13)
        p_sub.font.color.rgb = TEXT_MUTED


def add_card(slide, left, top, width, height, border_color=CARD_BORDER, bg_color=SURFACE_CARD):
    """Draws a rounded rectangular card container."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1.2)
    return card


# ===========================================================================
# SLIDE BUILDERS (10 SLIDES)
# ===========================================================================

def build_slide_1(prs):
    """SLIDE 1: Title Slide"""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    add_dark_background(slide)

    # Decorative Badge
    badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.3), Inches(4.5), Inches(0.45))
    badge.fill.solid()
    badge.fill.fore_color.rgb = RGBColor(30, 41, 59)
    badge.line.color.rgb = PRIMARY_BLUE
    p_badge = badge.text_frame.paragraphs[0]
    p_badge.text = "DEEP LEARNING & COMPUTER VISION PROJECT"
    p_badge.font.size = Pt(10)
    p_badge.font.bold = True
    p_badge.font.color.rgb = PRIMARY_BLUE
    p_badge.alignment = PP_ALIGN.CENTER

    # Main Project Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.9), Inches(11.7), Inches(1.6))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p1 = tf_title.paragraphs[0]
    p1.text = "AI-Based Road Damage Detection"
    p1.font.size = Pt(38)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_WHITE

    p2 = tf_title.add_paragraph()
    p2.text = "Automated Road Surface Distress Classification Using Deep Convolutional Neural Networks (CNN)"
    p2.font.size = Pt(18)
    p2.font.color.rgb = CYAN_ACCENT

    # Tagline
    tagline_box = slide.shapes.add_textbox(Inches(0.8), Inches(3.7), Inches(10.5), Inches(0.8))
    tf_tag = tagline_box.text_frame
    tf_tag.word_wrap = True
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = "Empowering civil infrastructure maintenance and traffic safety with automated, real-time computer vision and interactive web analytics."
    p_tag.font.size = Pt(14)
    p_tag.font.color.rgb = TEXT_MUTED

    # Bottom Metadata Card
    add_card(slide, 0.8, 4.8, 11.7, 1.8, border_color=PRIMARY_BLUE)
    meta_box = slide.shapes.add_textbox(Inches(1.1), Inches(4.95), Inches(11.1), Inches(1.5))
    tf_meta = meta_box.text_frame
    tf_meta.word_wrap = True
    
    p_m1 = tf_meta.paragraphs[0]
    p_m1.text = "PROJECT METADATA & ARTIFACTS"
    p_m1.font.size = Pt(11)
    p_m1.font.bold = True
    p_m1.font.color.rgb = PRIMARY_BLUE

    p_m2 = tf_meta.add_paragraph()
    p_m2.text = "• Student Developer: Gaius Parasa  |  College Mini Project\n" \
                "• Tech Stack: Python 3.10, TensorFlow, Keras CNN, Scikit-Learn, Streamlit\n" \
                "• Live Demo URL: https://road-damage-detection.streamlit.app\n" \
                "• GitHub Repository: https://github.com/gaiusparasa10-hash/Road-Damage-Detection"
    p_m2.font.size = Pt(12)
    p_m2.font.color.rgb = TEXT_WHITE


def build_slide_2(prs):
    """SLIDE 2: The Problem (4 Cards Grid)"""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    add_dark_background(slide)
    add_header(slide, "THE PROBLEM", "The Infrastructure Crisis Nobody Talks About",
               "Traditional roadway monitoring methods are dangerous, expensive, and unable to scale.")

    cards_data = [
        ("🚨 Reactive Maintenance", "Critical Delay",
         "Potholes and structural fissures are typically repaired only after vehicular damage, accidents, or public outcry.",
         RED_ACCENT),
        ("💸 Exorbitant Survey Costs", "Financial Burden",
         "Specialized laser profiler survey vans cost $100k-$250k+, putting regular road audits out of reach for most local municipalities.",
         AMBER_ACCENT),
        ("⏳ Hazardous Manual Audits", "Physical Risk",
         "Survey personnel physically walking live highways face extreme vehicular risks, with inspections taking weeks per municipal zone.",
         PURPLE_ACCENT),
        ("📉 Subjective & Fragmented Data", "Inconsistency",
         "Manual observer ratings vary significantly between inspectors, leading to inconsistent defect tracking and lost records.",
         CYAN_ACCENT),
    ]

    coords = [
        (0.8, 2.0, 5.7, 2.3),
        (6.8, 2.0, 5.7, 2.3),
        (0.8, 4.6, 5.7, 2.3),
        (6.8, 4.6, 5.7, 2.3),
    ]

    for (title, tag, desc, color), (x, y, w, h) in zip(cards_data, coords):
        add_card(slide, x, y, w, h, border_color=color)
        tbox = slide.shapes.add_textbox(Inches(x + 0.3), Inches(y + 0.2), Inches(w - 0.6), Inches(h - 0.4))
        tf = tbox.text_frame
        tf.word_wrap = True
        
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(16)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_WHITE
        
        p_tag = tf.add_paragraph()
        p_tag.text = f"KEY CHALLENGE: {tag.upper()}"
        p_tag.font.size = Pt(10)
        p_tag.font.bold = True
        p_tag.font.color.rgb = color

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = TEXT_MUTED


def build_slide_3(prs):
    """SLIDE 3: Our Solution"""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    add_dark_background(slide)
    add_header(slide, "OUR SOLUTION", "AI-Powered Road Damage Detection System",
               "A fast, automated deep learning vision pipeline that classifies road photos in under 50 milliseconds.")

    # Left Column: Features
    features = [
        ("🎯 Binary Classification Engine",
         "Separates roadway images into 'Normal Road' and 'Damaged Road' with high precision, removing subjective human bias."),
        ("🔍 Comprehensive Damage Taxonomy",
         "Trained on RDD2022 to detect longitudinal cracks, transverse cracks, alligator cracking, and hazardous potholes."),
        ("📊 Instant Confidence & Repair Insights",
         "Outputs exact percentage confidence along with actionable civil engineering recommendations for road crews."),
        ("🌐 100% Accessible Streamlit Web Portal",
         "Allows any municipal worker or student to upload an image from smartphone or dashcam without complex software setup.")
    ]

    y_start = 2.0
    for title, desc in features:
        add_card(slide, 0.8, y_start, 6.2, 1.15, border_color=PRIMARY_BLUE)
        tbox = slide.shapes.add_textbox(Inches(1.0), Inches(y_start + 0.1), Inches(5.8), Inches(0.95))
        tf = tbox.text_frame
        tf.word_wrap = True
        
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(13)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_WHITE
        
        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = TEXT_MUTED
        y_start += 1.3

    # Right Column: Visual Preview Card or Screenshot
    add_card(slide, 7.3, 2.0, 5.2, 4.95, border_color=CYAN_ACCENT)
    if os.path.exists(PREVIEW_IMG):
        slide.shapes.add_picture(PREVIEW_IMG, Inches(7.45), Inches(2.15), width=Inches(4.9))
    else:
        mock_box = slide.shapes.add_textbox(Inches(7.5), Inches(3.0), Inches(4.8), Inches(3.0))
        tf_m = mock_box.text_frame
        tf_m.word_wrap = True
        p_m = tf_m.paragraphs[0]
        p_m.text = "🛣️ Interactive Dashboard Preview\n\n• Live Image Upload (JPG/PNG)\n• Dual Class Detection\n• Confidence Score Gauge\n• Civil Repair Advice"
        p_m.font.size = Pt(14)
        p_m.font.color.rgb = TEXT_WHITE


def build_slide_4(prs):
    """SLIDE 4: System Architecture (4-Tier Pipeline)"""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    add_dark_background(slide)
    add_header(slide, "ARCHITECTURE", "4-Tier Deep Learning Architecture",
               "A clean, modular client-server-model pipeline engineered for simplicity and high performance.")

    tiers = [
        ("Layer 1: Data Pipeline", "RDD2022 dataset, Pascal VOC XML parser, filter corrupted files, 128×128 RGB normalizer.", PRIMARY_BLUE),
        ("Layer 2: CNN Core", "3 Conv2D blocks (32, 64, 128 filters), MaxPooling2D, Flatten, Dense(128), Dropout(0.5).", PURPLE_ACCENT),
        ("Layer 3: Evaluation Engine", "Binary crossentropy, Adam optimizer, confusion matrix, precision/recall/F1 metrics.", CYAN_ACCENT),
        ("Layer 4: Frontend UI", "Streamlit reactive web application, CLI inference script (predict.py), confidence meter.", GREEN_ACCENT)
    ]

    x_start = 0.8
    card_w = 2.75
    for title, desc, color in tiers:
        add_card(slide, x_start, 2.0, card_w, 3.4, border_color=color)
        tbox = slide.shapes.add_textbox(Inches(x_start + 0.15), Inches(2.15), Inches(card_w - 0.3), Inches(3.1))
        tf = tbox.text_frame
        tf.word_wrap = True
        
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = color
        
        p_d = tf.add_paragraph()
        p_d.text = f"\n{desc}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = TEXT_WHITE
        
        x_start += 2.95

    # Bottom Pipeline Summary
    add_card(slide, 0.8, 5.65, 11.7, 1.3, border_color=PRIMARY_BLUE)
    pbox = slide.shapes.add_textbox(Inches(1.0), Inches(5.75), Inches(11.3), Inches(1.1))
    tf_p = pbox.text_frame
    tf_p.word_wrap = True
    p_sum = tf_p.paragraphs[0]
    p_sum.text = "PIPELINE FLOW: Raw Road Image ➔ 128×128 RGB Preprocessing ➔ Rescaling [0, 1] ➔ 3-Block CNN Feature Extraction ➔ Dropout Regularization ➔ Sigmoid Binary Classifier ➔ Real-Time Web Alert"
    p_sum.font.size = Pt(12)
    p_sum.font.bold = True
    p_sum.font.color.rgb = CYAN_ACCENT


def build_slide_5(prs):
    """SLIDE 5: Core Pillars / Key Features"""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    add_dark_background(slide)
    add_header(slide, "CORE CAPABILITIES", "Four Pillars of the Detection System",
               "Engineered with foundational principles that guarantee real-world usability.")

    pillars = [
        ("🔍 Hierarchical Feature Learning",
         "Layer 1 extracts basic edges; Layer 2 recognizes crack contours; Layer 3 captures complex pothole cavities without hand-crafted features.",
         PRIMARY_BLUE),
        ("⚡ Ultra-Fast Edge Inference",
         "The lightweight 3.3M parameter model runs in <50ms per frame, making it ideal for future deployment on edge devices like Raspberry Pi or mobile.",
         CYAN_ACCENT),
        ("📊 Transparent Evaluation",
         "Incorporates true confusion matrix logging, training/validation loss graphs, and strict metric calculation without fabricated values.",
         GREEN_ACCENT),
        ("🛠️ Zero-Setup Web Usability",
         "Features both an interactive drag-and-drop Streamlit frontend and a terminal-based CLI script (`predict.py`) for automated pipelines.",
         AMBER_ACCENT)
    ]

    coords = [
        (0.8, 2.0, 5.7, 2.3),
        (6.8, 2.0, 5.7, 2.3),
        (0.8, 4.6, 5.7, 2.3),
        (6.8, 4.6, 5.7, 2.3),
    ]

    for (title, desc, color), (x, y, w, h) in zip(pillars, coords):
        add_card(slide, x, y, w, h, border_color=color)
        tbox = slide.shapes.add_textbox(Inches(x + 0.3), Inches(y + 0.25), Inches(w - 0.6), Inches(h - 0.5))
        tf = tbox.text_frame
        tf.word_wrap = True
        
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(16)
        p_t.font.bold = True
        p_t.font.color.rgb = color
        
        p_d = tf.add_paragraph()
        p_d.text = f"\n{desc}"
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = TEXT_MUTED


def build_slide_6(prs):
    """SLIDE 6: AI Workflow / Step-by-Step Flow"""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    add_dark_background(slide)
    add_header(slide, "AI WORKFLOW", "End-to-End Image Processing & Prediction Flow",
               "Every road photo travels through a structured deep learning sequence.")

    steps = [
        ("1. Image Upload", "User uploads road photo (JPG/PNG) via web UI or CLI tool.", PRIMARY_BLUE),
        ("2. Standardize", "Image converted to 3-channel RGB and resized to 128×128 pixels.", CYAN_ACCENT),
        ("3. Normalization", "Rescaling layer normalizes integer pixel values [0, 255] to [0.0, 1.0].", PURPLE_ACCENT),
        ("4. Conv2D Blocks", "3 stages of Convolution & MaxPooling learn spatial distress patterns.", PRIMARY_BLUE),
        ("5. Dense & Dropout", "128-neuron dense layer with 50% dropout prevents memorization.", AMBER_ACCENT),
        ("6. Classification", "Sigmoid neuron outputs probability: Normal Road vs. Damaged Road.", GREEN_ACCENT)
    ]

    card_w = 1.85
    gap = 0.12
    x = 0.8
    for title, desc, color in steps:
        add_card(slide, x, 2.0, card_w, 4.9, border_color=color)
        tbox = slide.shapes.add_textbox(Inches(x + 0.1), Inches(2.2), Inches(card_w - 0.2), Inches(4.5))
        tf = tbox.text_frame
        tf.word_wrap = True
        
        p_step = tf.paragraphs[0]
        p_step.text = title
        p_step.font.size = Pt(13)
        p_step.font.bold = True
        p_step.font.color.rgb = color
        
        p_desc = tf.add_paragraph()
        p_desc.text = f"\n{desc}"
        p_desc.font.size = Pt(11)
        p_desc.font.color.rgb = TEXT_WHITE
        
        x += card_w + gap


def build_slide_7(prs):
    """SLIDE 7: Tech Stack (6 Categorized Cards)"""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    add_dark_background(slide)
    add_header(slide, "TECH STACK", "Built With Modern Open-Source AI Technologies",
               "Carefully selected lightweight, battle-tested Python libraries.")

    stack_cards = [
        ("🧠 Deep Learning Framework", "TensorFlow 2.x & Keras\n• Sequential CNN API\n• Rescaling, Conv2D, Dropout\n• Adam Optimizer & Binary Crossentropy", PRIMARY_BLUE),
        ("🔢 Scientific Computing", "NumPy & SciPy\n• Tensor transformations\n• Array matrix manipulation\n• High-performance vectorization", CYAN_ACCENT),
        ("🖼️ Computer Vision & I/O", "Pillow (PIL) & Matplotlib\n• Lanczos high-res resampling\n• RGB color space conversion\n• Accuracy & Loss curve plotting", PURPLE_ACCENT),
        ("📈 Model Evaluation", "Scikit-Learn\n• Precision, Recall, F1 calculation\n• Confusion Matrix computation\n• Stratified data partitioning", GREEN_ACCENT),
        ("💻 Frontend & Web Hosting", "Streamlit & Uvicorn\n• Reactive Python web application\n• Instant cloud deployment\n• Drag-and-drop file uploader", AMBER_ACCENT),
        ("🚀 Version Control & CI", "Git & GitHub\n• Open-source codebase\n• GitHub-ready repository\n• Live demo integration", RED_ACCENT)
    ]

    coords = [
        (0.8, 2.0, 3.7, 2.3),
        (4.8, 2.0, 3.7, 2.3),
        (8.8, 2.0, 3.7, 2.3),
        (0.8, 4.6, 3.7, 2.3),
        (4.8, 4.6, 3.7, 2.3),
        (8.8, 4.6, 3.7, 2.3),
    ]

    for (title, desc, color), (x, y, w, h) in zip(stack_cards, coords):
        add_card(slide, x, y, w, h, border_color=color)
        tbox = slide.shapes.add_textbox(Inches(x + 0.2), Inches(y + 0.15), Inches(w - 0.4), Inches(h - 0.3))
        tf = tbox.text_frame
        tf.word_wrap = True
        
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(13)
        p_t.font.bold = True
        p_t.font.color.rgb = color
        
        p_d = tf.add_paragraph()
        p_d.text = f"\n{desc}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = TEXT_WHITE


def build_slide_8(prs):
    """SLIDE 8: Lessons Learned / Challenges & Solutions"""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    add_dark_background(slide)
    add_header(slide, "LESSONS LEARNED", "Challenges & How We Solved Them",
               "Real-world machine learning hurdles addressed through disciplined software engineering.")

    challenges = [
        ("⚠️ Challenge 1: Corrupted & Missing Annotations",
         "The raw RDD2022 dataset contained various images with missing or corrupted XML files.",
         "🛠️ Solution: Strict Preprocessing Filter",
         "Engineered preprocess.py to strictly validate XML nodes. Only images with verified 0-damage annotations are classed as Normal. Unlabeled files are safely skipped.",
         RED_ACCENT),
        ("⚖️ Challenge 2: Severe Class Imbalance",
         "Public road datasets focus primarily on damaged roads, leading to severe class skew against normal roads.",
         "🛠️ Solution: Dual Intake & Stratified Splitting",
         "Implemented dedicated normal road intake (data/raw/normal_images/) and 80/20 train-val splitting with fixed random seeds for reproducible balance.",
         AMBER_ACCENT),
        ("🛡️ Challenge 3: Risk of Overfitting",
         "Small initial datasets risk memorization rather than learning generalizable distress contours.",
         "🛠️ Solution: Architectural Regularization",
         "Embedded a 0.5 Dropout layer, 3 max-pooling stages, and input rescaling to enforce robust, generalized edge and texture detection.",
         GREEN_ACCENT)
    ]

    y_pos = 2.0
    for c_title, c_desc, s_title, s_desc, color in challenges:
        add_card(slide, 0.8, y_pos, 11.7, 1.5, border_color=color)
        tbox = slide.shapes.add_textbox(Inches(1.0), Inches(y_pos + 0.1), Inches(11.3), Inches(1.3))
        tf = tbox.text_frame
        tf.word_wrap = True
        
        p1 = tf.paragraphs[0]
        p1.text = f"{c_title}: {c_desc}"
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_WHITE
        
        p2 = tf.add_paragraph()
        p2.text = f"{s_title}: {s_desc}"
        p2.font.size = Pt(11.5)
        p2.font.color.rgb = color
        
        y_pos += 1.65


def build_slide_9(prs):
    """SLIDE 9: Results & Roadmap"""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    add_dark_background(slide)
    add_header(slide, "IMPACT & ROADMAP", "Results, Impact & What's Next",
               "Demonstrated project outcomes and strategic future expansion.")

    # Left Column: Delivered Results
    add_card(slide, 0.8, 2.0, 5.7, 4.95, border_color=PRIMARY_BLUE)
    tbox_l = slide.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.3), Inches(4.6))
    tf_l = tbox_l.text_frame
    tf_l.word_wrap = True
    
    p_lh = tf_l.paragraphs[0]
    p_lh.text = "🚀 DELIVERED RESULTS & IMPACT"
    p_lh.font.size = Pt(15)
    p_lh.font.bold = True
    p_lh.font.color.rgb = PRIMARY_BLUE
    
    p_l1 = tf_l.add_paragraph()
    p_l1.text = "\n• Automated Image Classification\n  Instant binary classification between intact and damaged pavements.\n" \
                "• Complete End-to-End Pipeline\n  Fully functional preprocessing, model training, evaluation, and CLI tools.\n" \
                "• Empirical Verification\n  Confusion matrix and loss/accuracy plots automatically saved to results/.\n" \
                "• Live Cloud Deployment\n  Hosted web interface ready for live presentation and testing."
    p_l1.font.size = Pt(11.5)
    p_l1.font.color.rgb = TEXT_WHITE

    # Right Column: Future Roadmap
    add_card(slide, 6.8, 2.0, 5.7, 4.95, border_color=PURPLE_ACCENT)
    tbox_r = slide.shapes.add_textbox(Inches(7.0), Inches(2.15), Inches(5.3), Inches(4.6))
    tf_r = tbox_r.text_frame
    tf_r.word_wrap = True
    
    p_rh = tf_r.paragraphs[0]
    p_rh.text = "🔭 FUTURE SCOPE & ROADMAP"
    p_rh.font.size = Pt(15)
    p_rh.font.bold = True
    p_rh.font.color.rgb = PURPLE_ACCENT
    
    p_r1 = tf_r.add_paragraph()
    p_r1.text = "\n1. Phase 1 — Mobile & Dashcam Video Feed\n   Continuous real-time road inspection using vehicle-mounted cameras.\n" \
                "2. Phase 2 — Object Detection & Bounding Boxes\n   Upgrading to YOLOv8 for precise square-meter crack and pothole measurement.\n" \
                "3. Phase 3 — GPS Geotagging & Municipal Heatmaps\n   Automatic GIS plotting of road distress coordinates for public works teams.\n" \
                "4. Phase 4 — Edge TPU & Fleet Deployment\n   Quantization for low-power Raspberry Pi 5 & Nvidia Jetson devices."
    p_r1.font.size = Pt(11.5)
    p_r1.font.color.rgb = TEXT_WHITE


def build_slide_10(prs):
    """SLIDE 10: Conclusion"""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    add_dark_background(slide)
    add_header(slide, "CONCLUSION", "AI Road Damage Detection — Built for Safer Roads",
               "A complete, practical deep learning project bridging computer vision with civil infrastructure.")

    # 3 Summary Cards
    cards = [
        ("🎯 Problem Solved", "Replaces hazardous, expensive manual road audits with automated, instant visual inference.", PRIMARY_BLUE),
        ("⚡ Technical Depth", "Constructed from scratch using TensorFlow, Keras CNN, Scikit-Learn metrics, and Streamlit.", CYAN_ACCENT),
        ("🌍 Real-World Impact", "Democratizes roadway health monitoring for municipal engineers and smart-city planning.", GREEN_ACCENT)
    ]

    x = 0.8
    for title, desc, color in cards:
        add_card(slide, x, 2.0, 3.7, 2.1, border_color=color)
        tbox = slide.shapes.add_textbox(Inches(x + 0.2), Inches(2.15), Inches(3.3), Inches(1.8))
        tf = tbox.text_frame
        tf.word_wrap = True
        
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = color
        
        p_d = tf.add_paragraph()
        p_d.text = f"\n{desc}"
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = TEXT_WHITE
        
        x += 4.0

    # Closing Quote Box
    add_card(slide, 0.8, 4.35, 11.7, 1.2, border_color=PURPLE_ACCENT)
    qbox = slide.shapes.add_textbox(Inches(1.0), Inches(4.45), Inches(11.3), Inches(1.0))
    tf_q = qbox.text_frame
    tf_q.word_wrap = True
    p_q = tf_q.paragraphs[0]
    p_q.text = '"Intelligent infrastructure monitoring is not just a research concept — it is the practical foundation for safer roads and smarter cities."'
    p_q.font.size = Pt(13)
    p_q.font.bold = True
    p_q.font.color.rgb = TEXT_ACCENT
    p_q.alignment = PP_ALIGN.CENTER

    # Thank you & Links
    tbox_end = slide.shapes.add_textbox(Inches(0.8), Inches(5.75), Inches(11.7), Inches(1.2))
    tf_end = tbox_end.text_frame
    tf_end.word_wrap = True
    p_end = tf_end.paragraphs[0]
    p_end.text = "Thank you! We welcome your questions and feedback.\n" \
                 "Live Demo: https://road-damage-detection.streamlit.app  |  GitHub: https://github.com/gaiusparasa10-hash/Road-Damage-Detection"
    p_end.font.size = Pt(13)
    p_end.font.bold = True
    p_end.font.color.rgb = TEXT_WHITE
    p_end.alignment = PP_ALIGN.CENTER


def main():
    print("=" * 70)
    print(" GENERATING 10-SLIDE PROFESSIONAL COLLEGE PRESENTATION (.PPTX)")
    print("=" * 70)

    prs = init_presentation()
    build_slide_1(prs)
    build_slide_2(prs)
    build_slide_3(prs)
    build_slide_4(prs)
    build_slide_5(prs)
    build_slide_6(prs)
    build_slide_7(prs)
    build_slide_8(prs)
    build_slide_9(prs)
    build_slide_10(prs)

    prs.save(OUTPUT_PPTX)
    print(f"[SUCCESS] Presentation generated: {OUTPUT_PPTX}")
    print("=" * 70)


if __name__ == "__main__":
    main()
