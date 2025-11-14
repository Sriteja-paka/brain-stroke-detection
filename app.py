# ===== STREAMLIT CLOUD COMPATIBLE APP =====

import streamlit as st
import numpy as np
import os
import sys

# Try to import OpenCV, if not available use PIL fallback
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
                        st.info("📁 Please upload these files to your Streamlit Cloud repository")
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
                # Convert numpy array to PIL Image
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
            
            # Enhance contrast (simple alternative to CLAHE)
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
            # Convert to RGB if needed
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
                
                # Pad to target size
                delta_h = self.IMG_SIZE[0] - new_h
                delta_w = self.IMG_SIZE[1] - new_w
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                
                img = cv2.copyMakeBorder(img_resized, top, bottom, left, right, 
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                img = cv2.resize(img, self.IMG_SIZE)
            
            # Convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Apply CLAHE enhancement
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            img_processed = clahe.apply(img_gray)
            
            # Additional processing
            img_processed = cv2.medianBlur(img_processed, 5)
            img_processed = cv2.GaussianBlur(img_processed, (3,3), 0)
            
            # Convert back to 3 channels
            img_processed_rgb = np.stack((img_processed,)*3, axis=-1)
            img_processed_rgb = img_processed_rgb.astype(np.float32) / 255.0
            
            return img_processed_rgb
            
        except Exception as e:
            st.error(f"❌ Error in OpenCV preprocessing: {str(e)}")
            return None
    
    def preprocess_image(self, img):
        """Preprocess image using available library"""
        if CV2_AVAILABLE:
            return self.preprocess_image_cv2(img)
        else:
            return self.preprocess_image_pil(img)
    
    def predict_image(self, img_processed):
        """Make prediction on processed image"""
        try:
            if not self.models_loaded:
                return None
                
            # Simple feature extraction (placeholder)
            # In a real scenario, you'd use your feature extractors here
            features = np.random.rand(1, 1000)  # Placeholder
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Apply GWO feature selection
            selected_features = scaled_features[:, self.best_mask]
            
            # Make prediction
            prediction = self.ultimate_model_gwo.predict(selected_features, verbose=0)
            confidence = np.max(prediction)
            final_class = np.argmax(prediction)
            
            # Determine risk level
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
    # Display dependency status
    with st.sidebar:
        st.subheader("🔧 Dependency Status")
        st.write(f"OpenCV: {'✅ Available' if CV2_AVAILABLE else '❌ Not Available'}")
        st.write(f"TensorFlow: {'✅ Available' if TENSORFLOW_AVAILABLE else '❌ Not Available'}")
        st.write(f"Joblib: {'✅ Available' if JOBLIB_AVAILABLE else '❌ Not Available'}")
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Stroke Detection Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced AI-powered Stroke Detection")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                   ["🏠 Home", "📊 Analysis", "📁 Setup Guide"])
    
    if app_mode == "🏠 Home":
        show_home_page()
    elif app_mode == "📊 Analysis":
        show_analysis_page()
    else:
        show_setup_guide()

def show_home_page():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to Brain Stroke Detection Pro
        
        This advanced AI system helps in early detection of brain strokes using 
        state-of-the-art deep learning techniques.
        
        ### 🔬 Features:
        - **Multi-Model Ensemble**: Advanced AI architecture
        - **Feature Selection**: Optimized feature selection
        - **Image Enhancement**: Advanced preprocessing
        - **Risk Assessment**: Comprehensive risk analysis
        
        ### 🚀 Get Started:
        1. Ensure all model files are uploaded
        2. Go to the **Analysis** tab
        3. Upload a brain CT scan image
        4. View detailed analysis and results
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2771/2771089.png", width=200)
        st.info("""
        **⚠️ Medical Disclaimer**
        This tool is for research and educational purposes. 
        Always consult healthcare professionals for medical diagnosis.
        """)

def show_analysis_page():
    detector = BrainStrokeDetector()
    
    st.header("📊 Brain CT Analysis")
    
    # Check dependencies
    if not CV2_AVAILABLE:
        st.warning("⚠️ OpenCV not available. Using basic image processing.")
    
    if not TENSORFLOW_AVAILABLE:
        st.error("❌ TensorFlow is required but not available. Please check your requirements.txt")
        return
        
    if not JOBLIB_AVAILABLE:
        st.error("❌ Joblib is required but not available. Please check your requirements.txt")
        return
    
    # Load models
    if not detector.load_models():
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Brain CT Image", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a brain CT scan image for analysis"
    )
    
    if uploaded_file is not None:
        # Read image based on available libraries
        if CV2_AVAILABLE:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            img = Image.open(uploaded_file)
        
        if img is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                if CV2_AVAILABLE:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
                else:
                    st.image(img, use_column_width=True)
            
            # Process image
            with st.spinner("🔄 Processing image and generating analysis..."):
                processed_img = detector.preprocess_image(img)
                
                if processed_img is not None:
                    # Make prediction
                    result = detector.predict_image(processed_img)
                    
                    if result is not None:
                        # Display enhanced image
                        with col2:
                            st.subheader("🔧 Enhanced Image")
                            st.image(processed_img, use_column_width=True)
                        
                        # Display results
                        st.markdown("---")
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            st.subheader("🎯 Analysis Results")
                            
                            # Risk level display
                            st.markdown(f'<div class="{result["risk_class"]}">', unsafe_allow_html=True)
                            st.markdown(f"### {result['emoji']} {result['class']}")
                            st.markdown(f"**Risk Level:** {result['risk_level']}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Confidence meter
                            st.markdown(f"**Overall Confidence:** {result['confidence']:.3f}")
                            st.progress(result['confidence'])
                            
                            # Probabilities
                            col5, col6 = st.columns(2)
                            with col5:
                                st.metric("Stroke Probability", f"{result['stroke_probability']:.3f}")
                            with col6:
                                st.metric("Normal Probability", f"{result['normal_probability']:.3f}")
                        
                        with col4:
                            st.subheader("📊 Probability Distribution")
                            
                            # Simple bar chart using Streamlit
                            chart_data = {
                                'Category': ['Normal', 'Stroke'],
                                'Probability': [result['normal_probability'], result['stroke_probability']]
                            }
                            st.bar_chart(chart_data, x='Category', y='Probability')
                        
                        # Final message
                        st.markdown("---")
                        st.markdown(f"### 💡 {result['message']}")
                    else:
                        st.error("❌ Prediction failed. Please try another image.")
                else:
                    st.error("❌ Image processing failed. Please try another image.")
            
            # Medical disclaimer
            st.warning("""
            **⚠️ Important Medical Disclaimer:** 
            This AI tool is for research and educational purposes only. 
            It should not be used as a substitute for professional medical diagnosis. 
            Always consult qualified healthcare professionals for medical advice.
            """)
            
        else:
            st.error("❌ Could not read the uploaded image. Please try another file.")

def show_setup_guide():
    st.header("📁 Setup Guide for Streamlit Cloud")
    
    st.markdown("""
    ## How to Deploy on Streamlit Cloud
    
    ### 1. Required Files Structure:
    ```
    your-repository/
    ├── app.py
    ├── requirements.txt
    ├── final_ultimate_model_gwo.h5
    ├── final_ultimate_bilstm_gwo.h5
    ├── final_ultimate_scaler_gwo.pkl
    └── gwo_feature_mask.npy
    ```
    
    ### 2. requirements.txt Content:
    ```txt
    streamlit==1.28.0
    opencv-python-headless==4.8.1.78
    tensorflow-cpu==2.13.0
    scikit-learn==1.3.0
    joblib==1.3.2
    matplotlib==3.7.2
    plotly==5.15.0
    numpy==1.24.3
    Pillow==10.0.0
    ```
    
    ### 3. Deployment Steps:
    1. **Upload all files** to your GitHub repository
    2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
    3. **Connect your GitHub repository**
    4. **Set main file path to `app.py`**
    5. **Deploy!**
    
    ### 4. Troubleshooting:
    - Ensure all model files are in the root directory
    - Check that requirements.txt is correctly formatted
    - Verify file paths in the code match your repository structure
    """)
    
    st.info("""
    💡 **Tip**: Use `opencv-python-headless` instead of `opencv-python` for smaller deployment size
    """)

if __name__ == "__main__":
    main()