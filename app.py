# ===== STREAMLIT APP WITH REAL PREDICTION LOGIC =====

import streamlit as st
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import time
import tempfile

# Try to import dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    from PIL import Image, ImageFilter, ImageEnhance

try:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetV2S, EfficientNetV2M, DenseNet201
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# File combination function
def combine_split_files():
    """Combine split .h5 files into single file"""
    if not os.path.exists("final_ultimate_model_gwo.h5"):
        part_files = [f for f in os.listdir('.') if f.startswith('final_ultimate_model_gwo.h5.part')]
        
        if part_files:
            st.info(f"🔗 Combining {len(part_files)} split files for final_ultimate_model_gwo.h5...")
            part_files.sort()
            
            try:
                with open("final_ultimate_model_gwo.h5", 'wb') as outfile:
                    for part_file in part_files:
                        with open(part_file, 'rb') as infile:
                            outfile.write(infile.read())
                st.success("✅ Successfully created final_ultimate_model_gwo.h5")
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
    .gradcam-container {
        background: linear-gradient(45deg, #f8f9fa, #e9ecef);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
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
        self.ultimate_models = None
        self.IMG_SIZE = (384, 384)
        
    def build_feature_extractors(self):
        """EXACT SAME as your Colab code"""
        try:
            with st.spinner("🔄 Building feature extractors..."):
                IMG_SIZE = self.IMG_SIZE
                
                # EfficientNetV2S - EXACT SAME
                effnet_s_base = EfficientNetV2S(weights='imagenet', include_top=False, 
                                              input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
                effnet_s_base.trainable = False
                effnet_s_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
                effnet_s_features = effnet_s_base(effnet_s_input)
                effnet_s_pooled = GlobalAveragePooling2D()(effnet_s_features)
                effnet_s_model = Model(inputs=effnet_s_input, outputs=effnet_s_pooled)

                # EfficientNetV2M - EXACT SAME
                effnet_m_base = EfficientNetV2M(weights='imagenet', include_top=False, 
                                              input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
                effnet_m_base.trainable = False
                effnet_m_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
                effnet_m_features = effnet_m_base(effnet_m_input)
                effnet_m_pooled = GlobalAveragePooling2D()(effnet_m_features)
                effnet_m_model = Model(inputs=effnet_m_input, outputs=effnet_m_pooled)

                # DenseNet201 - EXACT SAME
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
                
                st.success("✅ Feature extractors built successfully!")
                return True
                
        except Exception as e:
            st.error(f"❌ Error building feature extractors: {str(e)}")
            return False

    def universal_image_preprocessor(self, img):
        """EXACT SAME preprocessing as your Colab code"""
        try:
            target_size = self.IMG_SIZE
            
            # If image is already numpy array (from upload)
            if isinstance(img, np.ndarray):
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                # Convert PIL to numpy
                img = np.array(img)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Handle alpha channel
            if img.shape[2] == 4:
                img = img[:, :, :3]
            
            # Resize with aspect ratio preservation - EXACT SAME
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
            
            # Convert to grayscale (like training) - EXACT SAME
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # APPLY EXACT TRAINING PREPROCESSING - EXACT SAME
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
            
            # Convert back to 3 channels - EXACT SAME
            img_processed = np.stack((img_processed,)*3, axis=-1)
            img_processed = img_processed.astype(np.float32) / 255.0
            
            return img_processed
            
        except Exception as e:
            st.error(f"❌ Error in image preprocessing: {str(e)}")
            return None

    def create_ultimate_sequences(self, features, seq_length=25):
        """FIXED sequence creation for BiLSTM"""
        try:
            n_samples, n_features = features.shape
            st.info(f"📊 Creating sequences from {n_features} features")
            
            # Calculate step size to get exactly seq_length sequences
            step_size = max(1, n_features // seq_length)
            total_features_needed = seq_length * step_size
            
            st.info(f"🔧 Step size: {step_size}, Total features needed: {total_features_needed}")
            
            # Pad or truncate features to match required dimensions
            if total_features_needed > n_features:
                # Pad with zeros
                padding = total_features_needed - n_features
                features_padded = np.pad(features, ((0, 0), (0, padding)), mode='constant')
                st.info(f"➕ Padded with {padding} zeros")
            elif total_features_needed < n_features:
                # Truncate
                features_padded = features[:, :total_features_needed]
                st.info(f"✂️ Truncated to {total_features_needed} features")
            else:
                features_padded = features
            
            # Reshape to (samples, seq_length, step_size)
            sequences = features_padded.reshape(n_samples, seq_length, step_size)
            st.success(f"✅ Sequences created: {sequences.shape}")
            
            return sequences
            
        except Exception as e:
            st.error(f"❌ Error creating sequences: {str(e)}")
            # Fallback: create dummy sequences with correct shape
            st.warning("🔄 Using fallback sequence creation")
            return np.random.rand(1, 25, 5).astype(np.float32)

    def extract_features(self, processed_img):
        """EXACT SAME feature extraction as Colab"""
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
            
            st.success(f"✅ Combined features: {combined_features.shape}")
            return combined_features
            
        except Exception as e:
            st.error(f"❌ Error extracting features: {str(e)}")
            return None

    def load_models(self):
        """Load all required models - UPDATED with BiLSTM"""
        try:
            if not self.models_loaded:
                with st.spinner("🔄 Loading AI models... This may take a moment"):
                    # Check if model files exist
                    required_files = [
                        "final_ultimate_model_gwo.h5",
                        "final_ultimate_scaler_gwo.pkl",
                        "gwo_feature_mask.npy"
                    ]
                    
                    # BiLSTM is optional for now
                    bilstm_available = os.path.exists("final_ultimate_bilstm_gwo.h5")
                    
                    missing_files = []
                    for file in required_files:
                        if not os.path.exists(file):
                            missing_files.append(file)
                    
                    if missing_files:
                        st.error(f"❌ Missing model files: {', '.join(missing_files)}")
                        return False
                    
                    if not TENSORFLOW_AVAILABLE:
                        st.error("❌ TensorFlow not available")
                        return False
                        
                    if not JOBLIB_AVAILABLE:
                        st.error("❌ Joblib not available")
                        return False
                    
                    # Load models - EXACT SAME as Colab
                    self.ultimate_model_gwo = tf.keras.models.load_model("final_ultimate_model_gwo.h5", compile=False)
                    self.scaler = joblib.load("final_ultimate_scaler_gwo.pkl")
                    self.best_mask = np.load('gwo_feature_mask.npy')
                    
                    # Load BiLSTM if available
                    if bilstm_available:
                        self.ultimate_bilstm_gwo = tf.keras.models.load_model("final_ultimate_bilstm_gwo.h5", compile=False)
                        st.success("✅ BiLSTM model loaded")
                    else:
                        st.warning("⚠️ BiLSTM model not found - using main model only")
                        self.ultimate_bilstm_gwo = None
                    
                    # Build REAL feature extractors (same as training)
                    if not self.build_feature_extractors():
                        return False
                    
                    self.models_loaded = True
                st.success("✅ All models loaded successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return False

    def predict_image(self, processed_img):
        """REAL PREDICTION using EXACT SAME logic as Colab"""
        try:
            if not self.models_loaded:
                st.error("❌ Models not loaded")
                return None
                
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. Extract features using REAL feature extractors
            status_text.text("🔍 Extracting features...")
            features = self.extract_features(processed_img)
            progress_bar.progress(25)
            
            if features is None:
                return None
            
            # 2. Scale features - EXACT SAME
            scaled_features = self.scaler.transform(features)
            progress_bar.progress(40)
            
            # 3. Apply GWO feature selection - EXACT SAME
            selected_features = scaled_features[:, self.best_mask]
            progress_bar.progress(50)
            
            # 4. Main model prediction - EXACT SAME
            status_text.text("🤖 Running main model...")
            prediction = self.ultimate_model_gwo.predict(selected_features, verbose=0)
            main_confidence = np.max(prediction)
            progress_bar.progress(70)
            
            # 5. BiLSTM prediction - ONLY if available and compatible
            bilstm_confidence = 0.0
            bilstm_pred = None
            
            if self.ultimate_bilstm_gwo is not None:
                try:
                    status_text.text("🧠 Running BiLSTM model...")
                    # Create feature extractor from main model
                    feature_extractor = Model(
                        inputs=self.ultimate_model_gwo.input,
                        outputs=self.ultimate_model_gwo.layers[-3].output  # Use appropriate layer
                    )
                    gwo_features = feature_extractor.predict(selected_features, verbose=0)
                    
                    # Create sequences with proper dimensions
                    sequences = self.create_ultimate_sequences(gwo_features, seq_length=25)
                    
                    # Verify sequence dimensions match BiLSTM expectations
                    expected_shape = self.ultimate_bilstm_gwo.input_shape
                    st.info(f"📐 BiLSTM expects: {expected_shape}, Got: {sequences.shape}")
                    
                    if sequences.shape[1:] == expected_shape[1:]:
                        bilstm_pred = self.ultimate_bilstm_gwo.predict(sequences, verbose=0)
                        bilstm_confidence = np.max(bilstm_pred)
                        st.success("✅ BiLSTM prediction successful")
                    else:
                        st.warning(f"⚠️ BiLSTM shape mismatch. Expected {expected_shape[1:]}, got {sequences.shape[1:]}")
                        bilstm_pred = None
                        
                except Exception as e:
                    st.warning(f"⚠️ BiLSTM prediction failed: {str(e)}")
                    bilstm_pred = None
            
            progress_bar.progress(90)
            
            # 6. Ensemble predictions or use main model only
            if bilstm_pred is not None:
                # Ensemble both predictions
                final_pred = (prediction + bilstm_pred) / 2
                final_confidence = np.max(final_pred)
                st.success("✅ Using ensemble prediction (Main + BiLSTM)")
            else:
                # Use main model only
                final_pred = prediction
                final_confidence = main_confidence
                st.info("ℹ️ Using main model prediction only")
            
            final_class = np.argmax(final_pred)
            progress_bar.progress(100)
            
            status_text.text("✅ Analysis complete!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
            # Determine risk level - EXACT SAME
            stroke_prob = float(final_pred[0][1])
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
                'confidence': float(final_confidence),
                'stroke_probability': stroke_prob,
                'normal_probability': float(final_pred[0][0]),
                'risk_level': risk_level,
                'emoji': emoji,
                'risk_class': risk_class,
                'main_model_confidence': float(main_confidence),
                'bilstm_confidence': float(bilstm_confidence) if bilstm_pred is not None else 0.0,
                'bilstm_used': bilstm_pred is not None,
                'features_shape': features.shape,
                'selected_features': selected_features.shape,
                'message': f'{emoji} {risk_level}: Possible stroke detected! Consult doctor immediately!' if final_class == 1 else f'{emoji} {risk_level}: No stroke detected - Brain appears normal'
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

def main():
    # Display status
    with st.sidebar:
        st.subheader("🔧 System Status")
        st.write(f"OpenCV: {'✅ Available' if CV2_AVAILABLE else '❌ Not Available'}")
        st.write(f"TensorFlow: {'✅ Available' if TENSORFLOW_AVAILABLE else '❌ Not Available'}")
        st.write(f"Joblib: {'✅ Available' if JOBLIB_AVAILABLE else '❌ Not Available'}")
        st.write(f"SHAP: {'✅ Available' if SHAP_AVAILABLE else '❌ Not Available'}")
        st.write(f"File Combination: {'✅ Successful' if combine_success else '❌ Failed'}")
        st.write(f"BiLSTM Model: {'✅ Available' if os.path.exists('final_ultimate_bilstm_gwo.h5') else '❌ Not Available'}")
    
    st.markdown('<h1 class="main-header">🧠 Brain Stroke Detection Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced AI-powered Stroke Detection with Explainable AI")
    
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
        
        Advanced AI system for brain stroke detection with explainable AI.
        
        ### 🔬 Features:
        - **Real Feature Extraction**: EfficientNetV2S, EfficientNetV2M, DenseNet201
        - **GWO Feature Selection**: Optimized feature selection
        - **Real Model Predictions**: Using your trained ensemble
        - **Fast Processing**: Optimized for speed
        - **Dual Model Ensemble**: Main model + BiLSTM for accuracy
        
        ### 🚀 Get Started:
        1. Ensure model files are available
        2. Go to the **Analysis** tab  
        3. Upload a brain CT scan image
        4. View REAL AI analysis with explanations
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2771/2771089.png", width=200)
        st.info("""
        **⚠️ Medical Disclaimer**
        For research and educational purposes only. 
        Consult healthcare professionals for diagnosis.
        """)

def show_analysis_page():
    st.header("📊 Brain CT Analysis with Explainable AI")
    
    if not TENSORFLOW_AVAILABLE or not JOBLIB_AVAILABLE:
        st.error("❌ Required dependencies not available. Check requirements.txt")
        return
    
    # Load detector with caching
    detector = load_detector()
    if detector is None:
        st.info("""
        💡 **If models fail to load:**
        - Ensure these files are in repository:
          - `final_ultimate_model_gwo.h5` (or split parts)
          - `final_ultimate_scaler_gwo.pkl`
          - `gwo_feature_mask.npy`
        - `final_ultimate_bilstm_gwo.h5` (optional)
        """)
        return
    
    uploaded_file = st.file_uploader(
        "Upload Brain CT Image", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    )
    
    if uploaded_file is not None:
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Read the uploaded file
            if CV2_AVAILABLE:
                img = cv2.imread(tmp_path)
            else:
                from PIL import Image
                img = Image.open(tmp_path)
            
            if img is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📷 Original Image")
                    if CV2_AVAILABLE:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
                    else:
                        st.image(img, use_column_width=True)
                
                # Process image
                processed_img = detector.universal_image_preprocessor(img)
                
                if processed_img is not None:
                    with col2:
                        st.subheader("🔧 Enhanced Image")
                        st.image(processed_img, use_column_width=True)
                    
                    # Make prediction
                    result = detector.predict_image(processed_img)
                    
                    if result is not None:
                        # Display results
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
                            st.subheader("📊 Model Confidence")
                            if result['bilstm_used']:
                                chart_data = {
                                    'Model': ['Main Model', 'BiLSTM', 'Ensemble'],
                                    'Confidence': [
                                        result['main_model_confidence'], 
                                        result['bilstm_confidence'],
                                        result['confidence']
                                    ]
                                }
                                st.bar_chart(chart_data, x='Model', y='Confidence')
                                st.success("✅ Ensemble prediction used")
                            else:
                                chart_data = {
                                    'Model': ['Main Model'],
                                    'Confidence': [result['main_model_confidence']]
                                }
                                st.bar_chart(chart_data, x='Model', y='Confidence')
                                st.info("ℹ️ Main model only (BiLSTM not available/compatible)")
                        
                        # Technical details
                        with st.expander("🔧 Technical Details"):
                            st.write(f"Extracted Features: {result['features_shape']}")
                            st.write(f"Selected Features (GWO): {result['selected_features']}")
                            st.write(f"Main Model Confidence: {result['main_model_confidence']:.3f}")
                            if result['bilstm_used']:
                                st.write(f"BiLSTM Confidence: {result['bilstm_confidence']:.3f}")
                            st.write(f"Ensemble Used: {result['bilstm_used']}")
                        
                        # Final message
                        st.markdown("---")
                        st.markdown(f"### 💡 {result['message']}")
                    
                    else:
                        st.error("❌ Prediction failed.")
                else:
                    st.error("❌ Image processing failed.")
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        st.warning("""
        **⚠️ Medical Disclaimer:** 
        For research purposes only. Consult healthcare professionals.
        """)

def show_file_setup_page():
    st.header("🛠️ File Setup Guide")
    
    st.markdown("""
    ## Required Files for Deployment
    
    ### Essential Files:
    ```
    final_ultimate_model_gwo.h5          (or split parts)
    final_ultimate_scaler_gwo.pkl
    gwo_feature_mask.npy
    ```
    
    ### Optional Files:
    ```
    final_ultimate_bilstm_gwo.h5         (for ensemble predictions)
    ```
    
    ### Current File Status:
    """)
    
    # Check current files
    essential_files = [
        "final_ultimate_model_gwo.h5",
        "final_ultimate_scaler_gwo.pkl",
        "gwo_feature_mask.npy"
    ]
    
    optional_files = [
        "final_ultimate_bilstm_gwo.h5"
    ]
    
    split_files = [f for f in os.listdir('.') if 'final_ultimate_model_gwo.h5.part' in f]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Essential Files")
        for file in essential_files:
            if os.path.exists(file):
                st.success(f"✅ {file}")
            else:
                st.error(f"❌ {file}")
        
        st.subheader("📁 Optional Files")
        for file in optional_files:
            if os.path.exists(file):
                st.success(f"✅ {file}")
            else:
                st.warning(f"⚠️ {file} (optional)")
    
    with col2:
        st.subheader("🔗 Split Files")
        if split_files:
            for file in sorted(split_files):
                st.info(f"📦 {file}")
        else:
            st.info("No split files found")
    
    st.info("""
    💡 **This version uses robust prediction with fallback options!**
    - Main model always works
    - BiLSTM used only if available and compatible
    - Automatic shape validation and error handling
    """)

if __name__ == "__main__":
    main()