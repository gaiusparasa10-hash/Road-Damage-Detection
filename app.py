"""
=============================================================================
Project: AI-Based Road Damage Detection Using Deep Learning (CNN)
File: app.py
Description:
    Streamlit interactive web application for college demonstration.
    Allows users to upload road images (JPG, JPEG, PNG) and receive real-time
    binary classification:
    - NORMAL ROAD
    - DAMAGED ROAD
    Displays confidence score, model architecture details, and actual
    evaluation metrics.
=============================================================================
"""

import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Road Damage Detection System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# CONSTANTS & PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "road_damage_model.keras")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_JSON_PATH = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
ACCURACY_PLOT_PATH = os.path.join(RESULTS_DIR, "accuracy_plot.png")
LOSS_PLOT_PATH = os.path.join(RESULTS_DIR, "loss_plot.png")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")

IMAGE_SIZE = (128, 128)

# ---------------------------------------------------------------------------
# CUSTOM CSS STYLING FOR MODERN PRESENTATION
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .result-damaged {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border: 2px solid #EF4444;
        color: #991B1B;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 15px;
    }
    .result-normal {
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        border: 2px solid #22C55E;
        color: #166534;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 15px;
    }
    .result-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
    }
    .result-subtitle {
        font-size: 1.05rem;
        font-weight: 500;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MODEL LOADER WITH STREAMLIT CACHING
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_model():
    """
    Loads the trained Keras model into memory once and caches it.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as error:
        st.error(f"Error loading model: {error}")
        return None


def preprocess_uploaded_image(uploaded_file):
    """
    Reads an uploaded image file, converts to RGB, and resizes to (128, 128).
    Returns original PIL Image and batch array for inference.
    """
    try:
        pil_image = Image.open(uploaded_file)
        rgb_image = pil_image.convert("RGB")
        resized_image = rgb_image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(resized_image, dtype=np.float32)
        batch_array = np.expand_dims(img_array, axis=0)
        return pil_image, batch_array
    except Exception as error:
        st.error(f"Error processing image file: {error}")
        return None, None


# ---------------------------------------------------------------------------
# SIDEBAR - MODEL SPECIFICATIONS & ACTUAL EVALUATION METRICS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Model Architecture")
    st.markdown("""
    - **Architecture**: Sequential CNN
    - **Input Dimensions**: 128 × 128 × 3 (RGB)
    - **Feature Extractors**: 
      - Conv2D (32, 3×3) + MaxPool (2×2)
      - Conv2D (64, 3×3) + MaxPool (2×2)
      - Conv2D (128, 3×3) + MaxPool (2×2)
    - **Dense Head**: 128 Neurons (ReLU)
    - **Regularization**: Dropout (0.5)
    - **Output Layer**: 1 Neuron (Sigmoid)
    - **Classes**: 2 (Normal / Damaged)
    - **Initial Epochs**: 10
    """)
    st.markdown("---")

    st.markdown("### 📈 Evaluation Metrics")
    if os.path.exists(METRICS_JSON_PATH):
        try:
            with open(METRICS_JSON_PATH, "r") as f:
                metrics = json.load(f)
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Val Accuracy", f"{metrics.get('validation_accuracy', 'N/A')}%")
                st.metric("Precision", f"{metrics.get('precision', 'N/A')}%")
            with c2:
                st.metric("Val Loss", f"{metrics.get('validation_loss', 'N/A')}")
                st.metric("Recall", f"{metrics.get('recall', 'N/A')}%")
            
            st.metric("Weighted F1-Score", f"{metrics.get('f1_score', 'N/A')}%")
            if metrics.get("training_accuracy") is not None:
                st.caption(f"Training Accuracy: {metrics['training_accuracy']}%")
        except Exception:
            st.info("Metrics file found but could not be parsed.")
    else:
        st.info("Actual evaluation metrics will appear here after running `python evaluate_model.py`.")

    st.markdown("---")
    st.caption("College Mini Project • Road Damage Detection using Deep Learning")


# ---------------------------------------------------------------------------
# MAIN INTERFACE - HEADER & SYSTEM OVERVIEW
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🛣️ AI-Based Road Damage Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated road condition assessment powered by a Convolutional Neural Network (CNN). '
    'Upload a road surface photo to identify whether the road is in normal condition or damaged by cracks and potholes.</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# CHECK MODEL AVAILABILITY
# ---------------------------------------------------------------------------
model = load_cached_model()

if model is None:
    st.warning("""
    **⚠️ Trained Model Not Found!**
    
    The trained weights file `models/road_damage_model.keras` has not been generated yet.
    
    **How to initialize and train the model:**
    1. Ensure raw images and XML annotations are placed in `data/raw/` (or run with sample data).
    2. Preprocess dataset: `python preprocess.py`
    3. Train the CNN model: `python train_model.py`
    4. Evaluate model: `python evaluate_model.py`
    5. Refresh this application page.
    
    *Per college project guidelines, fake predictions are strictly disabled when the model is missing.*
    """)
    st.stop()

# ---------------------------------------------------------------------------
# IMAGE UPLOAD SECTION
# ---------------------------------------------------------------------------
st.markdown("### 📤 Upload Road Image for Classification")
uploaded_file = st.file_uploader(
    "Choose an image file (JPG, JPEG, or PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photograph of a paved road or highway surface."
)

if uploaded_file is None:
    st.info("👆 Please upload a road surface image above to begin detection.")
else:
    # -----------------------------------------------------------------------
    # IMAGE PROCESSING & INFERENCE
    # -----------------------------------------------------------------------
    pil_image, batch_input = preprocess_uploaded_image(uploaded_file)

    if pil_image is not None and batch_input is not None:
        col_img, col_result = st.columns([1.1, 1.2], gap="large")

        with col_img:
            st.markdown("#### 📷 Image Preview")
            st.image(pil_image, caption=f"Uploaded File: {uploaded_file.name}", use_container_width=True)
            st.caption(f"Original Resolution: {pil_image.width} × {pil_image.height} px | Resized to: 128 × 128 px for CNN")

        with col_result:
            st.markdown("#### 🔍 Detection Result")
            with st.spinner("Analyzing road surface patterns..."):
                prediction = model.predict(batch_input, verbose=0)
                # Output sigmoid probability for class index 1 (Normal)
                normal_probability = float(prediction[0][0])

                if normal_probability >= 0.5:
                    predicted_class = "NORMAL ROAD"
                    confidence = normal_probability * 100.0
                    is_damaged = False
                else:
                    predicted_class = "DAMAGED ROAD"
                    confidence = (1.0 - normal_probability) * 100.0
                    is_damaged = True

            # Display Classification Badge
            if is_damaged:
                st.markdown(f"""
                <div class="result-damaged">
                    <div class="result-title">⚠️ DAMAGED ROAD</div>
                    <div class="result-subtitle">Surface distress detected (Cracks / Potholes)</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-normal">
                    <div class="result-title">✅ NORMAL ROAD</div>
                    <div class="result-subtitle">Clean & Undamaged Road Surface</div>
                </div>
                """, unsafe_allow_html=True)

            # Confidence Metric & Progress Bar
            st.markdown("##### 🎯 Prediction Confidence")
            st.progress(confidence / 100.0)
            st.markdown(f"**Confidence Score:** `{confidence:.2f}%`")

            # Technical Interpretation
            st.markdown("##### 📝 Inspection Details")
            if is_damaged:
                st.markdown("""
                - **Condition**: Structural deterioration detected on roadway surface.
                - **Potential Damage Types**: Longitudinal crack, transverse crack, alligator cracking, or pothole.
                - **Recommendation**: Flag for maintenance survey and timely asphalt resurfacing.
                """)
            else:
                st.markdown("""
                - **Condition**: Uniform asphalt/concrete pavement with no structural distress.
                - **Safety Assessment**: Safe for regular vehicular traffic flow.
                - **Recommendation**: Maintain regular periodic inspection schedule.
                """)

# ---------------------------------------------------------------------------
# BOTTOM TABS - DETAILED REPORTS & DOCUMENTATION
# ---------------------------------------------------------------------------
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Evaluation Visualizations", "🧠 CNN Architecture Flow", "📖 Dataset & Damage Classes"])

with tab1:
    st.markdown("#### Model Performance Visualizations")
    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists(ACCURACY_PLOT_PATH):
            st.image(ACCURACY_PLOT_PATH, caption="Training vs Validation Accuracy", use_container_width=True)
        else:
            st.info("Accuracy plot not yet generated. Run `python evaluate_model.py`.")
            
        if os.path.exists(CONFUSION_MATRIX_PATH):
            st.image(CONFUSION_MATRIX_PATH, caption="Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix not yet generated. Run `python evaluate_model.py`.")

    with col_b:
        if os.path.exists(LOSS_PLOT_PATH):
            st.image(LOSS_PLOT_PATH, caption="Training vs Validation Loss", use_container_width=True)
        else:
            st.info("Loss plot not yet generated. Run `python evaluate_model.py`.")

with tab2:
    st.markdown("#### Convolutional Neural Network (CNN) Pipeline")
    st.markdown("""
    The CNN automatically extracts spatial hierarchical features from raw pixel data:
    1. **Input Layer (128×128×3)**: Accepts RGB color road images.
    2. **Rescaling (1./255)**: Normalizes color values between 0.0 and 1.0 for numerical stability.
    3. **Conv2D Block 1 (32 Filters, 3×3)**: Detects basic edges, boundaries, and gradients.
    4. **MaxPooling2D (2×2)**: Downsamples to 64×64 while preserving prominent edge activations.
    5. **Conv2D Block 2 (64 Filters, 3×3)**: Identifies crack textures and continuous linear patterns.
    6. **MaxPooling2D (2×2)**: Downsamples to 32×32 spatial resolution.
    7. **Conv2D Block 3 (128 Filters, 3×3)**: Identifies complex pothole contours and interconnected alligator cracks.
    8. **MaxPooling2D (2×2)**: Downsamples to 16×16 spatial resolution.
    9. **Flatten & Dense (128 Neurons)**: Combines all spatial representations into high-level features.
    10. **Dropout (0.5)**: Randomly deactivates 50% of hidden connections during training to prevent overfitting.
    11. **Output (1 Neuron, Sigmoid)**: Produces binary classification probability.
    """)

with tab3:
    st.markdown("#### RDD2022 (Road Damage Dataset 2022) Damage Taxonomy")
    st.markdown("""
    | Damage Class | Technical Code | Description | Typical Real-World Cause |
    | :--- | :--- | :--- | :--- |
    | **Longitudinal Cracks** | `D00`, `D10` | Parallel cracks along traffic direction | Heavy axle load, wheel tracks, construction joints |
    | **Transverse Cracks** | `D01` / `D00` | Cracks perpendicular to traffic flow | Temperature shrinkage, asphalt binder hardening |
    | **Alligator Cracks** | `D20` | Interconnected multi-sided fatigue cracks | Structural base weakness, repeated fatigue loads |
    | **Potholes & Bumps** | `D40` | Bowl-shaped road surface depressions | Water infiltration freezing/thawing under vehicle stress |
    | **Normal Road** | None | Smooth pavement with zero distress | Regular maintenance, newly paved surface |
    """)
