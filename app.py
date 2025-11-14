# ===== STREAMLIT APP - WITH SHAP EXPLANATIONS =====

import streamlit as st
import numpy as np
import os
import time
import tempfile

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
    page_title="Brain Stroke Detection",
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
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        border: 2px solid #ff4444;
    }
    .risk-medium {
        background-color: #ffd93d;
        padding: 15px;
        border-radius: 10px;
        color: black;
        text-align: center;
        font-weight: bold;
        border: 2px solid #ffb300;
    }
    .risk-low {
        background-color: #6bcf7f;
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        border: 2px solid #4caf50;
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
        self.shap_explainer = None
        
    def build_feature_extractors(self):
        """Build feature extractors"""
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

    def preprocess_image_cv2(self, img):
        """Use the EXACT same preprocessing as Colab"""
        try:
            # Your exact Colab preprocessing code here
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Resize with aspect ratio preservation (same as Colab)
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
            
            # Convert to grayscale (like training)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # APPLY EXACT TRAINING PREPROCESSING (SAME AS COLAB)
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
            img_processed_rgb = np.stack((img_processed,)*3, axis=-1)
            img_processed_rgb = img_processed_rgb.astype(np.float32) / 255.0
            
            return img_processed_rgb
            
        except Exception as e:
            st.error(f"❌ Error in OpenCV preprocessing: {str(e)}")
            return None

    def extract_features(self, processed_img):
        """Feature extraction"""
        try:
            input_batch = np.expand_dims(processed_img, axis=0)
            
            features_dict = {}
            for model_name, feature_model in self.ultimate_models.items():
                features = feature_model.predict(input_batch, verbose=0)
                features_dict[model_name] = features
            
            # Combine features
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
                    
                    # Initialize SHAP explainer
                    if SHAP_AVAILABLE:
                        with st.spinner("🔄 Initializing SHAP explainer..."):
                            # Create a wrapper function for SHAP
                            def model_predict(x):
                                # Scale features
                                x_scaled = self.scaler.transform(x)
                                # Apply feature selection
                                x_selected = x_scaled[:, self.best_mask]
                                # Make prediction
                                return self.ultimate_model_gwo.predict(x_selected, verbose=0)
                            
                            # Create background data for SHAP
                            background_data = np.zeros((1, 1280 + 1280 + 1920))  # Adjust based on your feature size
                            self.shap_explainer = shap.Explainer(model_predict, background_data)
                    
                    self.models_loaded = True
                st.success("✅ All models loaded successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return False

    def predict_with_shap(self, processed_img):
        """Make prediction with SHAP explanations"""
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
            confidence = float(np.max(prediction))
            final_class = int(np.argmax(prediction))
            progress_bar.progress(85)
            
            # 5. Generate SHAP explanations
            shap_values = None
            if SHAP_AVAILABLE and self.shap_explainer is not None:
                status_text.text("📊 Generating SHAP explanations...")
                try:
                    shap_values = self.shap_explainer(features)
                    progress_bar.progress(95)
                except Exception as e:
                    st.warning(f"⚠️ SHAP explanation failed: {str(e)}")
                    shap_values = None
            
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
            
            # Result dictionary with SHAP values
            result = {
                'class': 'STROKE' if final_class == 1 else 'NORMAL',
                'confidence': confidence,
                'stroke_probability': stroke_prob,
                'normal_probability': normal_prob,
                'risk_level': risk_level,
                'emoji': emoji,
                'risk_class': risk_class,
                'shap_values': shap_values,
                'features': features
            }
            
            return result
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            return None

    def display_shap_explanation(self, result):
        """Display SHAP explanation if available"""
        if result.get('shap_values') is None or not SHAP_AVAILABLE:
            return
        
        try:
            st.markdown("---")
            st.subheader("🔍 SHAP Explanation")
            
            shap_values = result['shap_values']
            features = result['features']
            
            # Create SHAP summary plot
            st.write("**Feature Importance Summary:**")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values.values, features, show=False)
            st.pyplot(fig)
            plt.close()
            
            # Force plot for the current prediction
            st.write("**Prediction Explanation:**")
            shap_force = shap.force_plot(
                shap_values.base_values[0], 
                shap_values.values[0], 
                features[0],
                matplotlib=True,
                show=False
            )
            st.pyplot(shap_force)
            plt.close()
            
        except Exception as e:
            st.warning(f"⚠️ Could not display SHAP explanation: {str(e)}")

@st.cache_resource
def load_detector():
    """Cache the detector to avoid reloading models"""
    detector = BrainStrokeDetector()
    if detector.load_models():
        return detector
    return None

def show_home_page():
    st.markdown("""
    ## 🧠 Welcome to Brain Stroke Detection
    
    **Advanced AI system for brain stroke detection using CT scans.**
    
    ### 🔬 How It Works:
    1. **Upload** a brain CT scan image
    2. **AI Processing** with multiple feature extractors
    3. **Feature Selection** using Grey Wolf Optimizer
    4. **Risk Assessment** with probability scores
    5. **SHAP Explanations** for model interpretability
    
    ### 🚀 Get Started:
    Go to the **Analysis** tab to upload a brain CT image.
    """)

def show_analysis_page():
    st.header("📊 Brain CT Analysis")
    
    if not TENSORFLOW_AVAILABLE or not JOBLIB_AVAILABLE:
        st.error("❌ Required dependencies not available")
        return
    
    detector = load_detector()
    if detector is None:
        st.error("❌ Failed to load AI models. Please check the File Setup tab.")
        return
    
    if not SHAP_AVAILABLE:
        st.warning("⚠️ SHAP is not available. Install with: `pip install shap`")
    
    uploaded_file = st.file_uploader(
        "Upload Brain CT Image", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Read image
            if CV2_AVAILABLE:
                img = cv2.imread(tmp_path)
            else:
                from PIL import Image
                img = Image.open(tmp_path)
                img = np.array(img)
            
            if img is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📷 Original Image")
                    if CV2_AVAILABLE and len(img.shape) == 3:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
                    else:
                        st.image(img, use_column_width=True)
                
                # Process image using your exact preprocessing
                processed_img = detector.preprocess_image_cv2(img)
                
                if processed_img is not None:
                    with col2:
                        st.subheader("🔧 Enhanced Image")
                        st.image(processed_img, use_column_width=True)
                    
                    # Make prediction with SHAP
                    result = detector.predict_with_shap(processed_img)
                    
                    if result is not None:
                        # Display results
                        st.markdown("---")
                        
                        # Risk card
                        st.markdown(f'<div class="{result["risk_class"]}">', unsafe_allow_html=True)
                        st.markdown(f"## {result['emoji']} {result['risk_level']}")
                        st.markdown(f"### Diagnosis: {result['class']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Probabilities
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            st.subheader("📊 Probability Scores")
                            st.metric("Stroke Probability", f"{result['stroke_probability']:.3f}")
                            st.progress(result['stroke_probability'])
                            
                            st.metric("Normal Probability", f"{result['normal_probability']:.3f}")
                            st.progress(result['normal_probability'])
                        
                        with col4:
                            st.subheader("📈 Probability Chart")
                            chart_data = {
                                'Category': ['Normal', 'Stroke'],
                                'Probability': [
                                    result['normal_probability'], 
                                    result['stroke_probability']
                                ]
                            }
                            st.bar_chart(chart_data, x='Category', y='Probability')
                        
                        # Display SHAP explanations
                        detector.display_shap_explanation(result)
                        
                        # Final message
                        st.markdown("---")
                        if result['class'] == 'STROKE':
                            st.error(f"🚨 **Urgent Medical Attention Recommended** - Stroke probability: {result['stroke_probability']:.1%}")
                        else:
                            st.success(f"✅ **Normal Brain Patterns Detected** - Normal probability: {result['normal_probability']:.1%}")
                    
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        st.warning("**⚠️ Medical Disclaimer:** For research purposes only. Consult healthcare professionals.")

def show_file_setup_page():
    st.header("🛠️ File Setup")
    
    st.markdown("""
    ## Required Files:
    - `final_ultimate_model_gwo.h5`
    - `final_ultimate_scaler_gwo.pkl`  
    - `gwo_feature_mask.npy`
    """)
    
    # Check files
    files_to_check = [
        "final_ultimate_model_gwo.h5",
        "final_ultimate_scaler_gwo.pkl",
        "gwo_feature_mask.npy"
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            st.success(f"✅ {file}")
        else:
            st.error(f"❌ {file}")
    
    st.markdown("---")
    st.subheader("Dependencies Status")
    st.write(f"SHAP: {'✅ Available' if SHAP_AVAILABLE else '❌ Not Available'}")

def main():
    # Sidebar
    with st.sidebar:
        st.title("🧠 Navigation")
        app_mode = st.selectbox(
            "Choose Mode", 
            ["🏠 Home", "📊 Analysis", "🛠️ File Setup"]
        )
        
        st.markdown("---")
        st.subheader("System Status")
        st.write(f"OpenCV: {'✅' if CV2_AVAILABLE else '❌'}")
        st.write(f"TensorFlow: {'✅' if TENSORFLOW_AVAILABLE else '❌'}")
        st.write(f"Joblib: {'✅' if JOBLIB_AVAILABLE else '❌'}")
        st.write(f"SHAP: {'✅' if SHAP_AVAILABLE else '❌'}")
    
    # Main content
    st.markdown('<h1 class="main-header">🧠 Brain Stroke Detection</h1>', unsafe_allow_html=True)
    
    if app_mode == "🏠 Home":
        show_home_page()
    elif app_mode == "📊 Analysis":
        show_analysis_page()
    else:
        show_file_setup_page()

if __name__ == "__main__":
    main()