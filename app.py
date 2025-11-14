# ===== STREAMLIT APP - ULTRA CLEAN WORKING VERSION =====

import streamlit as st
import numpy as np
import os
import time
import tempfile
import matplotlib.pyplot as plt

# Try to import dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetV2S, EfficientNetV2M, DenseNet201
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D
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
    if not os.path.exists("final_ultimate_model_gwo.h5"):
        part_files = [f for f in os.listdir('.') if f.startswith('final_ultimate_model_gwo.h5.part')]
        
        if part_files:
            st.info(f"🔗 Combining {len(part_files)} split files...")
            part_files.sort()
            
            try:
                with open("final_ultimate_model_gwo.h5", 'wb') as outfile:
                    for part_file in part_files:
                        with open(part_file, 'rb') as infile:
                            outfile.write(infile.read())
                st.success("✅ Model file created successfully!")
                return True
            except Exception as e:
                st.error(f"❌ Error combining files: {str(e)}")
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
    .prediction-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class BrainStrokeDetector:
    def __init__(self):
        self.models_loaded = False
        self.ultimate_model_gwo = None
        self.scaler = None
        self.best_mask = None
        self.ultimate_models = None
        self.IMG_SIZE = (384, 384)
        
    def build_feature_extractors(self):
        """Build feature extractors - EXACT SAME as Colab code"""
        try:
            with st.spinner("🔄 Building feature extractors..."):
                IMG_SIZE = self.IMG_SIZE
                
                # EfficientNetV2S
                effnet_s_base = EfficientNetV2S(weights='imagenet', include_top=False, 
                                              input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
                effnet_s_base.trainable = False
                effnet_s_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
                effnet_s_features = effnet_s_base(effnet_s_input)
                effnet_s_pooled = GlobalAveragePooling2D()(effnet_s_features)
                effnet_s_model = Model(inputs=effnet_s_input, outputs=effnet_s_pooled)

                # EfficientNetV2M
                effnet_m_base = EfficientNetV2M(weights='imagenet', include_top=False, 
                                              input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
                effnet_m_base.trainable = False
                effnet_m_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
                effnet_m_features = effnet_m_base(effnet_m_input)
                effnet_m_pooled = GlobalAveragePooling2D()(effnet_m_features)
                effnet_m_model = Model(inputs=effnet_m_input, outputs=effnet_m_pooled)

                # DenseNet201
                densenet_base = DenseNet201(weights='imagenet', include_top=False, 
                                          input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
                densenet_base.trainable = False
                densenet_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
                densenet_features = densenet_base(densenet_input)
                densenet_pooled = GlobalAveragePooling2D()(densenet_features)
                densenet_model = Model(inputs=densenet_input, outputs=densenet_pooled)

                self.ultimate_models = {
                    'effnet_s': effnet_s_model,
                    'effnet_m': effnet_m_model,
                    'densenet': densenet_model
                }
                
                return True
                
        except Exception as e:
            st.error(f"❌ Error building feature extractors: {str(e)}")
            return False

    def universal_image_preprocessor(self, img):
        """Image preprocessing - EXACT SAME as Colab code"""
        try:
            target_size = self.IMG_SIZE
            
            # Convert to numpy array if needed
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            
            # Handle different image formats
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Handle alpha channel
            if img.shape[2] == 4:
                img = img[:, :, :3]
            
            # Resize with aspect ratio preservation
            h, w = img.shape[:2]
            if h != target_size[0] or w != target_size[1]:
                scale = min(target_size[0] / h, target_size[1] / w)
                new_h, new_w = int(h * scale), int(w * scale)
                img_resized = cv2.resize(img, (new_w, new_h))
                
                # Pad to target size
                delta_h = target_size[0] - new_h
                delta_w = target_size[1] - new_w
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                
                img = cv2.copyMakeBorder(img_resized, top, bottom, left, right, 
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                img = cv2.resize(img, target_size)
            
            # Convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # APPLY EXACT TRAINING PREPROCESSING
            clahe1 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            clahe2 = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(12,12))
            clahe3 = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(16,16))

            img1 = clahe1.apply(img_gray)
            img2 = clahe2.apply(img_gray)
            img3 = clahe3.apply(img_gray)

            img_temp = cv2.addWeighted(img1, 0.5, img2, 0.3, 0)
            img_processed = cv2.addWeighted(img_temp, 0.7, img3, 0.3, 0)

            img_processed = cv2.medianBlur(img_processed, 7)
            img_processed = cv2.GaussianBlur(img_processed, (5,5), 0)

            kernel1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            kernel2 = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])

            img_sharp1 = cv2.filter2D(img_processed, -1, kernel1)
            img_sharp2 = cv2.filter2D(img_processed, -1, kernel2)
            img_processed = cv2.addWeighted(img_sharp1, 0.6, img_sharp2, 0.4, 0)

            kernel = np.ones((3,3), np.uint8)
            img_processed = cv2.morphologyEx(img_processed, cv2.MORPH_CLOSE, kernel)
            img_processed = cv2.morphologyEx(img_processed, cv2.MORPH_OPEN, kernel)
            
            # Convert back to 3 channels
            img_processed = np.stack((img_processed,)*3, axis=-1)
            img_processed = img_processed.astype(np.float32) / 255.0
            
            return img_processed
            
        except Exception as e:
            st.error(f"❌ Error in image preprocessing: {str(e)}")
            return None

    def extract_features(self, processed_img):
        """Feature extraction - EXACT SAME as Colab"""
        try:
            input_batch = np.expand_dims(processed_img, axis=0)
            
            features_dict = {}
            for model_name, feature_model in self.ultimate_models.items():
                features = feature_model.predict(input_batch, verbose=0)
                features_dict[model_name] = features
            
            # Combine features EXACTLY like in training
            combined_features = np.concatenate([
                features_dict['effnet_s'],
                features_dict['effnet_m'], 
                features_dict['densenet']
            ], axis=1)
            
            return combined_features
            
        except Exception as e:
            st.error(f"❌ Error extracting features: {str(e)}")
            return None

    def load_models(self):
        """Load all required models"""
        try:
            if not self.models_loaded:
                with st.spinner("🔄 Loading AI models..."):
                    # Check if model files exist
                    required_files = [
                        "final_ultimate_model_gwo.h5",
                        "final_ultimate_scaler_gwo.pkl",
                        "gwo_feature_mask.npy"
                    ]
                    
                    missing_files = []
                    for file in required_files:
                        if not os.path.exists(file):
                            missing_files.append(file)
                    
                    if missing_files:
                        st.error(f"❌ Missing files: {', '.join(missing_files)}")
                        return False
                    
                    if not TENSORFLOW_AVAILABLE:
                        st.error("❌ TensorFlow not available")
                        return False
                        
                    if not JOBLIB_AVAILABLE:
                        st.error("❌ Joblib not available")
                        return False
                    
                    # Load models
                    self.ultimate_model_gwo = tf.keras.models.load_model("final_ultimate_model_gwo.h5", compile=False)
                    self.scaler = joblib.load("final_ultimate_scaler_gwo.pkl")
                    self.best_mask = np.load('gwo_feature_mask.npy')
                    
                    # Build feature extractors
                    if not self.build_feature_extractors():
                        return False
                    
                    self.models_loaded = True
                st.success("✅ All models loaded successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return False

    def predict_image(self, processed_img):
        """CLEAN PREDICTION - No Grad-CAM, No SHAP, No complex explanations"""
        try:
            if not self.models_loaded:
                st.error("❌ Models not loaded")
                return None
                
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. Extract features
            status_text.text("🔍 Extracting features...")
            features = self.extract_features(processed_img)
            progress_bar.progress(30)
            
            if features is None:
                st.error("❌ Feature extraction failed")
                return None
            
            # 2. Scale features
            scaled_features = self.scaler.transform(features)
            progress_bar.progress(50)
            
            # 3. Apply GWO feature selection
            selected_features = scaled_features[:, self.best_mask]
            progress_bar.progress(70)
            
            # 4. Make prediction
            status_text.text("🤖 Running AI model...")
            prediction = self.ultimate_model_gwo.predict(selected_features, verbose=0)
            confidence = np.max(prediction)
            final_class = np.argmax(prediction)
            progress_bar.progress(100)
            
            status_text.text("✅ Analysis complete!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
            # Get probabilities
            stroke_prob = float(prediction[0][1])
            normal_prob = float(prediction[0][0])
            
            # Determine risk level
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
            
            # SIMPLE result dictionary - NO COMPLEX EXPLANATIONS
            result = {
                'class': 'STROKE' if final_class == 1 else 'NORMAL',
                'confidence': float(confidence),
                'stroke_probability': stroke_prob,
                'normal_probability': normal_prob,
                'risk_level': risk_level,
                'emoji': emoji,
                'risk_class': risk_class
            }
            
            return result
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            return None

@st.cache_resource
def load_detector():
    """Cache the detector to avoid reloading models"""
    detector = BrainStrokeDetector()
    if detector.load_models():
        return detector
    return None

def show_home_page():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🧠 Welcome to Brain Stroke Detection Pro
        
        **Advanced AI system for brain stroke detection using CT scans.**
        
        ### 🔬 Features:
        - **Multi-Model Feature Extraction**: EfficientNetV2S, EfficientNetV2M, DenseNet201
        - **Grey Wolf Optimizer**: Advanced feature selection
        - **Real Model Predictions**: Using your trained ensemble model
        - **Medical Grade Processing**: Professional image enhancement
        
        ### 🚀 How to Use:
        1. Navigate to the **📊 Analysis** tab  
        2. Upload a brain CT scan image (JPG, PNG, etc.)
        3. View AI analysis results in seconds
        4. Get risk assessment and probabilities
        
        ### 📊 Output Includes:
        - Original vs. Enhanced image comparison
        - Stroke probability scores
        - Risk level classification
        - Confidence metrics
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2771/2771089.png", width=200)
        st.info("""
        **⚠️ Medical Disclaimer**
        
        This tool is for research and educational purposes only. 
        Always consult qualified healthcare professionals for medical diagnoses.
        Do not use this tool for emergency medical decisions.
        """)

def show_analysis_page():
    st.header("📊 Brain CT Scan Analysis")
    
    if not TENSORFLOW_AVAILABLE or not JOBLIB_AVAILABLE:
        st.error("❌ Required dependencies not available. Please install TensorFlow and Joblib.")
        st.code("pip install tensorflow joblib opencv-python")
        return
    
    # Load detector
    with st.spinner("🔄 Initializing AI system..."):
        detector = load_detector()
    
    if detector is None:
        st.error("""
        ❌ **Models failed to load**
        
        Please ensure these files are in your directory:
        - `final_ultimate_model_gwo.h5`
        - `final_ultimate_scaler_gwo.pkl`  
        - `gwo_feature_mask.npy`
        """)
        
        st.info("""
        💡 **Troubleshooting Tips:**
        - Check if all required files are present
        - Ensure TensorFlow is properly installed
        - Verify file permissions
        - Check the **🛠️ File Setup** tab for details
        """)
        return
    
    st.success("✅ AI system ready for analysis!")
    
    uploaded_file = st.file_uploader(
        "📁 Upload Brain CT Image", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a brain CT scan image for stroke detection analysis",
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Read and display image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                if CV2_AVAILABLE:
                    img = cv2.imread(tmp_path)
                    if img is not None:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
                    else:
                        st.error("❌ Could not read image file")
                        return
                else:
                    from PIL import Image
                    img = Image.open(tmp_path)
                    st.image(img, use_column_width=True)
                    img = np.array(img)
            
            # Process image
            with st.spinner("🔄 Processing image..."):
                processed_img = detector.universal_image_preprocessor(img)
            
            if processed_img is not None:
                with col2:
                    st.subheader("🔧 Enhanced Image")
                    st.image(processed_img, use_column_width=True)
                    st.caption("AI-enhanced version for better feature detection")
                
                # Make prediction
                result = detector.predict_image(processed_img)
                
                if result is not None:
                    # Display results - CLEAN AND SIMPLE
                    st.markdown("---")
                    
                    # Risk card
                    st.markdown(f'<div class="{result["risk_class"]}">', unsafe_allow_html=True)
                    col3, col4 = st.columns([2, 1])
                    with col3:
                        st.markdown(f"## {result['emoji']} {result['risk_level']}")
                        st.markdown(f"### Diagnosis: {result['class']}")
                    with col4:
                        st.markdown(f"**Confidence:** {result['confidence']:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Probabilities
                    col5, col6 = st.columns(2)
                    
                    with col5:
                        st.subheader("📊 Probability Scores")
                        st.metric(
                            "Stroke Probability", 
                            f"{result['stroke_probability']:.3f}",
                            delta=f"{(result['stroke_probability'] - 0.5):.3f}" if result['stroke_probability'] > 0.5 else None,
                            delta_color="inverse"
                        )
                        st.progress(result['stroke_probability'])
                        
                        st.metric(
                            "Normal Probability", 
                            f"{result['normal_probability']:.3f}",
                            delta=f"{(result['normal_probability'] - 0.5):.3f}" if result['normal_probability'] > 0.5 else None
                        )
                        st.progress(result['normal_probability'])
                    
                    with col6:
                        st.subheader("📈 Probability Chart")
                        chart_data = {
                            'Category': ['Normal', 'Stroke'],
                            'Probability': [
                                result['normal_probability'], 
                                result['stroke_probability']
                            ]
                        }
                        st.bar_chart(chart_data, x='Category', y='Probability', color='#ff4b4b')
                    
                    # Final recommendation
                    st.markdown("---")
                    if result['class'] == 'STROKE':
                        st.error(f"""
                        🚨 **Recommendation: URGENT MEDICAL ATTENTION NEEDED**
                        
                        The AI has detected signs consistent with stroke with **{result['stroke_probability']:.1%} probability**.
                        
                        **Immediate Actions:**
                        - Contact emergency services
                        - Do not delay medical consultation
                        - Follow healthcare professional guidance
                        """)
                    else:
                        st.success(f"""
                        ✅ **Recommendation: Routine Monitoring**
                        
                        The AI analysis shows normal brain patterns with **{result['normal_probability']:.1%} probability**.
                        
                        **Next Steps:**
                        - Continue regular health checkups
                        - Monitor for any new symptoms
                        - Maintain healthy lifestyle habits
                        """)
                    
                    # Technical details (collapsible)
                    with st.expander("🔧 Technical Details"):
                        st.write(f"- **Model:** final_ultimate_model_gwo")
                        st.write(f"- **Feature Selection:** GWO Optimized")
                        st.write(f"- **Confidence Score:** {result['confidence']:.3f}")
                        st.write(f"- **Final Prediction:** {result['class']}")
                        st.write(f"- **Risk Classification:** {result['risk_level']}")
                
                else:
                    st.error("❌ Prediction failed. Please try with a different image.")
            else:
                st.error("❌ Image processing failed. Please check the image format and try again.")
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        # Medical disclaimer
        st.warning("""
        **⚠️ IMPORTANT MEDICAL DISCLAIMER:** 
        This AI tool is for research and educational purposes only. It is NOT a substitute for professional medical diagnosis, 
        advice, or treatment. Always consult qualified healthcare professionals for medical concerns and emergencies.
        """)

def show_file_setup_page():
    st.header("🛠️ File Setup Guide")
    
    st.markdown("""
    ## Required Files for the Application
    
    Make sure these files are present in your working directory:
    """)
    
    # File status check
    files_to_check = [
        ("final_ultimate_model_gwo.h5", "Main trained model file"),
        ("final_ultimate_scaler_gwo.pkl", "Feature scaler for preprocessing"), 
        ("gwo_feature_mask.npy", "GWO feature selection mask")
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 File Status Check")
        all_files_ok = True
        
        for file_name, description in files_to_check:
            if os.path.exists(file_name):
                st.success(f"✅ **{file_name}**")
                st.caption(description)
            else:
                st.error(f"❌ **{file_name}**")
                st.caption(description)
                all_files_ok = False
        
        if all_files_ok:
            st.balloons()
            st.success("🎉 All files are present! You're ready to use the application.")
        else:
            st.error("❌ Some files are missing. Please check the setup instructions below.")
    
    with col2:
        st.subheader("🔗 Split File Detection")
        split_files = [f for f in os.listdir('.') if 'final_ultimate_model_gwo.h5.part' in f]
        
        if split_files:
            st.info(f"📦 Found {len(split_files)} split files")
            for file in sorted(split_files):
                st.code(file)
            
            if st.button("🔄 Combine Split Files Now"):
                if combine_split_files():
                    st.success("✅ Files combined successfully!")
                    st.rerun()
        else:
            st.info("No split model files detected")
    
    # Setup instructions
    st.markdown("---")
    st.subheader("📋 Setup Instructions")
    
    st.markdown("""
    ### If Files Are Missing:
    
    1. **Check your project directory** for the required files
    2. **Ensure proper file permissions**
    3. **Verify file integrity** - files should not be corrupted
    4. **For split files** - ensure all parts are present
    
    ### Expected File Sizes:
    - `final_ultimate_model_gwo.h5`: ~100-500MB (varies based on training)
    - `final_ultimate_scaler_gwo.pkl`: ~1-10MB  
    - `gwo_feature_mask.npy`: ~1-5MB
    
    ### Troubleshooting:
    - Restart the application after adding missing files
    - Check the console for detailed error messages
    - Ensure sufficient disk space is available
    - Verify TensorFlow and dependencies are properly installed
    """)

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2771/2771089.png", width=80)
        st.title("🧠 Stroke Detection")
        
        st.markdown("---")
        st.subheader("🔧 System Status")
        
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.write(f"OpenCV: {'✅' if CV2_AVAILABLE else '❌'}")
            st.write(f"TensorFlow: {'✅' if TENSORFLOW_AVAILABLE else '❌'}")
        with status_col2:
            st.write(f"Joblib: {'✅' if JOBLIB_AVAILABLE else '❌'}")
            st.write(f"Files: {'✅' if combine_success else '❌'}")
        
        st.markdown("---")
        
        # Navigation
        st.subheader("📍 Navigation")
        app_mode = st.radio(
            "Choose Section",
            ["🏠 Home", "📊 Analysis", "🛠️ File Setup"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("""
        **About This App:**
        
        Advanced AI system for brain stroke detection using ensemble deep learning and optimized feature selection.
        
        *For research purposes only.*
        """)
    
    # Main content area
    st.markdown('<h1 class="main-header">🧠 Brain Stroke Detection Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced AI-Powered Stroke Detection System")
    
    # Page routing
    if app_mode == "🏠 Home":
        show_home_page()
    elif app_mode == "📊 Analysis":
        show_analysis_page()
    else:
        show_file_setup_page()

if __name__ == "__main__":
    main()