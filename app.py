
import streamlit as st
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

def combine_split_files():
    """Combine the 7 specific part files (silent version)"""
    target_file = 'feature_extractor.keras'
    
    # Check if already combined
    if os.path.exists(target_file):
        return True
        
    # Define the exact 7 part files you have
    expected_parts = [
        'feature_extractor.keras.part000',
        'feature_extractor.keras.part001', 
        'feature_extractor.keras.part002',
        'feature_extractor.keras.part003',
        'feature_extractor.keras.part004',
        'feature_extractor.keras.part005',
        'feature_extractor.keras.part006'
    ]
    
    # Check if all parts exist
    for part in expected_parts:
        if not os.path.exists(part):
            st.error(f"❌ Missing model part: {part}")
            return False
    
    # Combine the parts silently
    try:
        with open(target_file, 'wb') as outfile:
            for part in expected_parts:
                with open(part, 'rb') as infile:
                    outfile.write(infile.read())
        return True
        
    except Exception as e:
        st.error(f"❌ Error combining model: {e}")
        return False

# Cache resources
@st.cache_resource
def load_artifacts():
    """Load all model artifacts (silent combination)"""
    # Combine split files first (no messages)
    if not combine_split_files():
        return None, None, None, None
        
    try:
        # Load all models
        model = load_model('best_bilstm.keras', compile=False)
        scaler = joblib.load('scaler_features.pkl')
        mask = np.load('best_mask.npy')
        feature_extractor = load_model('feature_extractor.keras', compile=False)
        
        return model, scaler, mask, feature_extractor
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None

# Load artifacts
model, scaler, mask, feature_extractor = load_artifacts()

if model is None:
    st.error("❌ Failed to load AI model. Please refresh the page or check the console for errors.")
    st.stop()

# Constants
IMG_SIZE = (224, 224)
SEQ_LEN = 16

def preprocess_image(image):
    """Preprocess uploaded image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, IMG_SIZE)
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized

def extract_features(img_array, feature_extractor):
    """Extract features using the feature extractor"""
    img_batch = np.expand_dims(img_array, axis=0)
    features = feature_extractor.predict(img_batch, verbose=0)
    return features

def reshape_for_lstm(Xs, seq_len=SEQ_LEN):
    """Reshape features for LSTM"""
    n, d = Xs.shape
    step = int(np.ceil(d / seq_len))
    total = step * seq_len
    if total > d:
        Xs = np.pad(Xs, ((0, 0), (0, total - d)), mode='constant')
    else:
        Xs = Xs[:, :total]
    return Xs.reshape(n, seq_len, step)

# Main UI - Clean version without combination messages
st.title("🧠 Brain Stroke Detection System")
st.markdown("Upload a CT scan image to detect potential stroke conditions")

# File upload
uploaded_file = st.file_uploader(
    "Choose CT scan image", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear CT scan image for analysis"
)

if uploaded_file is not None:
    # Read and display image using PIL
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded CT Scan", use_container_width=True)
    
    with col2:
        if st.button("🔍 Analyze for Stroke", type="primary", use_container_width=True):
            with st.spinner("Analyzing CT scan..."):
                try:
                    # Preprocess image
                    img_processed = preprocess_image(image)
                    
                    # Extract features using CNN
                    features = extract_features(img_processed, feature_extractor)
                    
                    # Scale features
                    feat_scaled = scaler.transform(features)
                    
                    # Apply feature selection mask
                    feat_sel = feat_scaled[:, mask == 1]
                    
                    # Reshape for LSTM
                    seq_input = reshape_for_lstm(feat_sel)
                    
                    # Predict
                    prediction = model.predict(seq_input, verbose=0)
                    class_idx = np.argmax(prediction, axis=1)[0]
                    confidence = np.max(prediction)
                    
                    # Display results
                    if class_idx == 1:
                        st.error(f"🟥 **Stroke Detected**")
                        st.warning(f"Confidence: {confidence*100:.2f}%")
                        st.info("⚠️ Please consult a healthcare professional immediately!")
                    else:
                        st.success(f"🟩 **Normal Brain**")
                        st.success(f"Confidence: {confidence*100:.2f}%")
                        st.info("✅ No signs of stroke detected")
                    
                    # Confidence meter
                    st.progress(float(confidence))
                    st.caption(f"Model confidence: {confidence*100:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

# Instructions
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. **Upload** a clear CT scan image (JPG, JPEG, or PNG format)
    2. **Click** the 'Analyze for Stroke' button
    3. **View** the results with confidence percentage
    4. **Important**: This is for educational purposes only
    5. **Always consult** healthcare professionals for medical diagnosis
    """)

# Footer
st.markdown("---")
st.caption("AI-Powered Stroke Detection System | For educational and research purposes")
