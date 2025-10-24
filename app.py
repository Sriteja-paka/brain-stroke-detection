import streamlit as st
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Brain Stroke Detection",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Cache resources
@st.cache_resource
def load_artifacts():
    """Load model and artifacts once"""
    try:
        model = load_model('best_bilstm.keras', compile=False)
        scaler = joblib.load('scaler_features.pkl')
        mask = np.load('best_mask.npy')
        feature_extractor = load_model('feature_extractor.keras', compile=False)
        return model, scaler, mask, feature_extractor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

# Load artifacts
model, scaler, mask, feature_extractor = load_artifacts()

if model is None:
    st.error("❌ Failed to load AI model. Please check if model files are available.")
    st.stop()

# Constants
IMG_SIZE = (224, 224)
SEQ_LEN = 16

def preprocess_image(image):
    """Preprocess uploaded image to match training pipeline"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale and apply CLAHE (same as training)
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_gray)
    img_enhanced = img_enhanced.astype(np.float32) / 255.0
    
    # Resize to match training
    img_resized = cv2.resize(img_enhanced, IMG_SIZE)
    
    # Convert to 3-channel (same as training)
    img_final = np.stack((img_resized,)*3, axis=-1)
    
    return img_final

def extract_features(img_array, feature_extractor):
    """Extract features using the CNN feature extractor"""
    # Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Extract features
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

# Main UI
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
                    
                    st.write(f"🔍 Features extracted: {features.shape}")
                    
                    # Scale features
                    feat_scaled = scaler.transform(features)
                    
                    # Apply feature selection mask
                    feat_sel = feat_scaled[:, mask == 1]
                    
                    st.write(f"🔍 After feature selection: {feat_sel.shape}")
                    
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
                    
                    # Show prediction probabilities
                    st.write(f"📊 Prediction probabilities: Normal: {prediction[0][0]:.3f}, Stroke: {prediction[0][1]:.3f}")
                    
                    # Confidence meter
                    st.progress(float(confidence))
                    st.caption(f"Model confidence: {confidence*100:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

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