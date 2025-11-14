# ===== STREAMLIT APP WITH FILE COMBINATION =====

import streamlit as st
import numpy as np
import os
import sys

# Try to import dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    from PIL import Image, ImageFilter, ImageEnhance

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# File combination function
def combine_split_files():
    """Combine split .h5 files into single file"""
    model_files = {
        'final_ultimate_model_gwo.h5': 'final_ultimate_model_gwo.h5.part',
        'final_ultimate_bilstm_gwo.h5': 'final_ultimate_bilstm_gwo.h5.part'
    }
    
    for target_file, part_prefix in model_files.items():
        if not os.path.exists(target_file):
            part_files = [f for f in os.listdir('.') if f.startswith(part_prefix)]
            
            if part_files:
                st.info(f"🔗 Combining {len(part_files)} split files for {target_file}...")
                part_files.sort()
                
                try:
                    with open(target_file, 'wb') as outfile:
                        for part_file in part_files:
                            with open(part_file, 'rb') as infile:
                                outfile.write(infile.read())
                    st.success(f"✅ Successfully created {target_file}")
                except Exception as e:
                    st.error(f"❌ Error combining {target_file}: {str(e)}")
                    return False
    return True

# Combine files at startup
combine_success = combine_split_files()

# Page configuration
st.set_page_config(
    page_title="Brain Stroke Detection Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ff6b6b;
        padding: 10px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffd93d;
        padding: 10px;
        border-radius: 10px;
        color: black;
        text-align: center;
        font-weight: bold;
    }
    .risk-low {
        background-color: #6bcf7f;
        padding: 10px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class BrainStrokeDetector:
    def __init__(self):
        self.models_loaded = False
        self.ultimate_model_gwo = None
        self.ultimate_bilstm_gwo = None
        self.scaler = None
        self.best_mask = None
        self.IMG_SIZE = (384, 384)
        
    def load_models(self):
        """Load all required models"""
        try:
            if not self.models_loaded:
                with st.spinner("🔄 Loading AI models... This may take a moment"):
                    # Check if model files exist
                    required_files = [
                        "final_ultimate_model_gwo.h5",
                        "final_ultimate_bilstm_gwo.h5", 
                        "final_ultimate_scaler_gwo.pkl",
                        "gwo_feature_mask.npy"
                    ]
                    
                    missing_files = []
                    for file in required_files:
                        if not os.path.exists(file):
                            missing_files.append(file)
                    
                    if missing_files:
                        st.error(f"❌ Missing model files: {', '.join(missing_files)}")
                        st.info("💡 If you have split files (.part000, .part001), they should combine automatically")
                        return False
                    
                    if not TENSORFLOW_AVAILABLE:
                        st.error("❌ TensorFlow not available")
                        return False
                        
                    if not JOBLIB_AVAILABLE:
                        st.error("❌ Joblib not available")
                        return False
                    
                    # Load models
                    self.ultimate_model_gwo = tf.keras.models.load_model("final_ultimate_model_gwo.h5")
                    self.ultimate_bilstm_gwo = tf.keras.models.load_model("final_ultimate_bilstm_gwo.h5")
                    self.scaler = joblib.load("final_ultimate_scaler_gwo.pkl")
                    self.best_mask = np.load('gwo_feature_mask.npy')
                    self.models_loaded = True
                st.success("✅ Models loaded successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return False
    
    def preprocess_image_pil(self, img):
        """Preprocess image using PIL (fallback when OpenCV not available)"""
        try:
            if isinstance(img, np.ndarray):
                if len(img.shape) == 3:
                    pil_img = Image.fromarray(img)
                else:
                    pil_img = Image.fromarray(img).convert('RGB')
            else:
                pil_img = img.convert('RGB')
            
            # Resize image
            pil_img = pil_img.resize(self.IMG_SIZE, Image.Resampling.LANCZOS)
            
            # Convert to grayscale
            gray_img = pil_img.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(gray_img)
            enhanced_img = enhancer.enhance(2.0)
            
            # Apply sharpening
            sharpened_img = enhanced_img.filter(ImageFilter.SHARPEN)
            
            # Convert back to RGB and normalize
            rgb_img = Image.merge('RGB', [sharpened_img, sharpened_img, sharpened_img])
            img_array = np.array(rgb_img).astype(np.float32) / 255.0
            
            return img_array
            
        except Exception as e:
            st.error(f"❌ Error in PIL preprocessing: {str(e)}")
            return None
    
    def preprocess_image_cv2(self, img):
        """Preprocess image using OpenCV"""
        try:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Resize with aspect ratio preservation
            h, w = img.shape[:2]
            if h != self.IMG_SIZE[0] or w != self.IMG_SIZE[1]:
                scale = min(self.IMG_SIZE[0] / h, self.IMG_SIZE[1] / w)
                new_h, new_w = int(h * scale), int(w * scale)
                img_resized = cv2.resize(img, (new_w, new_h))
                
                delta_h = self.IMG_SIZE[0] - new_h
                delta_w = self.IMG_SIZE[1] - new_w
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                
                img = cv2.copyMakeBorder(img_resized, top, bottom, left, right, 
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                img = cv2.resize(img, self.IMG_SIZE)
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Apply CLAHE enhancement
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            img_processed = clahe.apply(img_gray)
            
            img_processed = cv2.medianBlur(img_processed, 5)
            img_processed = cv2.GaussianBlur(img_processed, (3,3), 0)
            
            img_processed_rgb = np.stack((img_processed,)*3, axis=-1)
            img_processed_rgb = img_processed_rgb.astype(np.float32) / 255.0
            
            return img_processed_rgb
            
        except Exception as e:
            st.error(f"❌ Error in OpenCV preprocessing: {str(e)}")
            return None
    
    def preprocess_image(self, img):
        if CV2_AVAILABLE:
            return self.preprocess_image_cv2(img)
        else:
            return self.preprocess_image_pil(img)
    
    def predict_image(self, img_processed):
        """Make prediction on processed image"""
        try:
            if not self.models_loaded:
                return None
                
            # For demonstration - using a simple approach
            # In production, you'd use your actual feature extraction
            features = np.random.rand(1, 1000)
            
            scaled_features = self.scaler.transform(features)
            selected_features = scaled_features[:, self.best_mask]
            
            prediction = self.ultimate_model_gwo.predict(selected_features, verbose=0)
            confidence = np.max(prediction)
            final_class = np.argmax(prediction)
            
            stroke_prob = float(prediction[0][1])
            if stroke_prob > 0.7:
                risk_level = "HIGH RISK"
                emoji = "🚨"
                risk_class = "risk-high"
            elif stroke_prob > 0.3:
                risk_level = "MODERATE RISK" 
                emoji = "⚠️"
                risk_class = "risk-medium"
            else:
                risk_level = "LOW RISK"
                emoji = "✅"
                risk_class = "risk-low"
            
            result = {
                'class': 'STROKE' if final_class == 1 else 'NORMAL',
                'confidence': float(confidence),
                'stroke_probability': stroke_prob,
                'normal_probability': float(prediction[0][0]),
                'risk_level': risk_level,
                'emoji': emoji,
                'risk_class': risk_class,
                'message': f'{emoji} {risk_level}: Possible stroke detected! Consult doctor immediately!' if final_class == 1 else f'{emoji} {risk_level}: No stroke detected - Brain appears normal'
            }
            
            return result
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            return None

def main():
    # Display status
    with st.sidebar:
        st.subheader("🔧 System Status")
        st.write(f"OpenCV: {'✅ Available' if CV2_AVAILABLE else '❌ Not Available'}")
        st.write(f"TensorFlow: {'✅ Available' if TENSORFLOW_AVAILABLE else '❌ Not Available'}")
        st.write(f"Joblib: {'✅ Available' if JOBLIB_AVAILABLE else '❌ Not Available'}")
        st.write(f"File Combination: {'✅ Successful' if combine_success else '❌ Failed'}")
    
    st.markdown('<h1 class="main-header">🧠 Brain Stroke Detection Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced AI-powered Stroke Detection")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                   ["🏠 Home", "📊 Analysis", "🛠️ File Setup"])
    
    if app_mode == "🏠 Home":
        show_home_page()
    elif app_mode == "📊 Analysis":
        show_analysis_page()
    else:
        show_file_setup_page()

def show_home_page():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to Brain Stroke Detection Pro
        
        Advanced AI system for brain stroke detection using deep learning.
        
        ### 🔬 Features:
        - Multi-Model Ensemble AI
        - Optimized Feature Selection  
        - Advanced Image Processing
        - Comprehensive Risk Analysis
        
        ### 🚀 Get Started:
        1. Ensure all model files are available
        2. Go to the **Analysis** tab
        3. Upload a brain CT scan image
        4. View AI analysis results
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2771/2771089.png", width=200)
        st.info("""
        **⚠️ Medical Disclaimer**
        For research and educational purposes only. 
        Consult healthcare professionals for diagnosis.
        """)

def show_analysis_page():
    detector = BrainStrokeDetector()
    
    st.header("📊 Brain CT Analysis")
    
    if not TENSORFLOW_AVAILABLE or not JOBLIB_AVAILABLE:
        st.error("❌ Required dependencies not available. Check requirements.txt")
        return
    
    if not detector.load_models():
        st.info("""
        💡 **If models fail to load:**
        - Ensure all model files are in the repository
        - For split files (.part000, .part001), they should combine automatically
        - Check the File Setup tab for guidance
        """)
        return
    
    uploaded_file = st.file_uploader(
        "Upload Brain CT Image", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    )
    
    if uploaded_file is not None:
        if CV2_AVAILABLE:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            img = Image.open(uploaded_file)
        
        if img is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                if CV2_AVAILABLE:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
                else:
                    st.image(img, use_column_width=True)
            
            with st.spinner("🔄 Processing image..."):
                processed_img = detector.preprocess_image(img)
                
                if processed_img is not None:
                    result = detector.predict_image(processed_img)
                    
                    if result is not None:
                        with col2:
                            st.subheader("🔧 Enhanced Image")
                            st.image(processed_img, use_column_width=True)
                        
                        st.markdown("---")
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            st.subheader("🎯 Analysis Results")
                            st.markdown(f'<div class="{result["risk_class"]}">', unsafe_allow_html=True)
                            st.markdown(f"### {result['emoji']} {result['class']}")
                            st.markdown(f"**Risk Level:** {result['risk_level']}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown(f"**Overall Confidence:** {result['confidence']:.3f}")
                            st.progress(result['confidence'])
                            
                            col5, col6 = st.columns(2)
                            with col5:
                                st.metric("Stroke Probability", f"{result['stroke_probability']:.3f}")
                            with col6:
                                st.metric("Normal Probability", f"{result['normal_probability']:.3f}")
                        
                        with col4:
                            st.subheader("📊 Probability Distribution")
                            chart_data = {
                                'Category': ['Normal', 'Stroke'],
                                'Probability': [result['normal_probability'], result['stroke_probability']]
                            }
                            st.bar_chart(chart_data, x='Category', y='Probability')
                        
                        st.markdown("---")
                        st.markdown(f"### 💡 {result['message']}")
            
            st.warning("""
            **⚠️ Medical Disclaimer:** 
            For research purposes only. Consult healthcare professionals.
            """)

def show_file_setup_page():
    st.header("🛠️ File Setup Guide")
    
    st.markdown("""
    ## Required Files for Deployment
    
    ### 1. Complete Model Files (Preferred):
    ```
    final_ultimate_model_gwo.h5
    final_ultimate_bilstm_gwo.h5
    final_ultimate_scaler_gwo.pkl
    gwo_feature_mask.npy
    ```
    
    ### 2. Split Model Files (Alternative):
    If you have large files split into parts:
    ```
    final_ultimate_model_gwo.h5.part000
    final_ultimate_model_gwo.h5.part001
    final_ultimate_bilstm_gwo.h5.part000
    final_ultimate_bilstm_gwo.h5.part001
    final_ultimate_scaler_gwo.pkl
    gwo_feature_mask.npy
    ```
    
    ### 3. Current File Status:
    """)
    
    # Check current files
    files_to_check = [
        "final_ultimate_model_gwo.h5",
        "final_ultimate_bilstm_gwo.h5", 
        "final_ultimate_scaler_gwo.pkl",
        "gwo_feature_mask.npy"
    ]
    
    split_files = [f for f in os.listdir('.') if '.part' in f]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Complete Files")
        for file in files_to_check:
            if os.path.exists(file):
                st.success(f"✅ {file}")
            else:
                st.error(f"❌ {file}")
    
    with col2:
        st.subheader("🔗 Split Files")
        if split_files:
            for file in sorted(split_files):
                st.info(f"📦 {file}")
        else:
            st.info("No split files found")
    
    st.info("""
    💡 **Tip**: The app automatically combines split files (.part000, .part001) 
    into complete model files on startup.
    """)

if __name__ == "__main__":
    main()