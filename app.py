# ===== STREAMLIT APP - WORKING SHAP ONLY =====

import streamlit as st
import numpy as np
import os
import sys
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
        self.scaler = None
        self.best_mask = None
        self.ultimate_models = None
        self.IMG_SIZE = (384, 384)
        
    def build_feature_extractors(self):
        """EXACT SAME as your Colab code"""
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
        """EXACT SAME preprocessing as your Colab code"""
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

    def generate_shap_analysis(self, selected_features):
        """SIMPLE SHAP analysis without complex array operations"""
        try:
            if not SHAP_AVAILABLE:
                return None, None
            
            if selected_features is None or selected_features.size == 0:
                return None, None
            
            # Use very simple background
            background = np.random.rand(5, selected_features.shape[1]) * 0.1
            
            # Simple prediction function
            def predict_fn(x):
                return self.ultimate_model_gwo.predict(x, verbose=0)
            
            # Simple explainer
            explainer = shap.KernelExplainer(predict_fn, background)
            
            # Get SHAP values for stroke class (class 1)
            shap_values = explainer.shap_values(selected_features, nsamples=5)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class output - take stroke class (index 1)
                if len(shap_values) > 1:
                    stroke_shap = shap_values[1]  # Class 1 for stroke
                else:
                    stroke_shap = shap_values[0]
            else:
                # Single output
                stroke_shap = shap_values
            
            # Ensure we have 1D array for single prediction
            if hasattr(stroke_shap, 'shape') and len(stroke_shap.shape) > 1:
                stroke_shap = stroke_shap[0]  # Take first sample
            
            # Convert to numpy array safely
            shap_array = np.array(stroke_shap).flatten()
            
            # Get top 10 features by absolute impact
            top_n = min(10, len(shap_array))
            top_indices = np.argsort(np.abs(shap_array))[::-1][:top_n]
            top_features = [f"Feature_{i}" for i in top_indices]
            top_shap_values = shap_array[top_indices]
            
            # Calculate percentages for table
            total_impact = np.sum(np.abs(shap_array))
            if total_impact > 0:
                percentages = [(abs(val) / total_impact) * 100 for val in top_shap_values]
            else:
                percentages = [0] * len(top_shap_values)
            
            # Create the bar chart
            fig, ax = plt.subplots(figsize=(12, 8))
            y_pos = np.arange(len(top_features))
            
            # Color coding: green for positive (supports stroke), red for negative (against stroke)
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_shap_values]
            
            bars = ax.barh(y_pos, top_shap_values, color=colors, alpha=0.8, height=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features, fontsize=10)
            ax.set_xlabel('SHAP Value Impact', fontsize=12, fontweight='bold')
            ax.set_title('Top 10 Features Influencing Stroke Prediction', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, top_shap_values)):
                width = bar.get_width()
                label_x_pos = width + (0.01 if width >= 0 else -0.01)
                ha = 'left' if width >= 0 else 'right'
                color = 'darkgreen' if width >= 0 else 'darkred'
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                       f'{value:.4f}', ha=ha, va='center', fontweight='bold', color=color)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', label='Supports Stroke Prediction'),
                Patch(facecolor='#e74c3c', label='Against Stroke Prediction')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            
            plt.tight_layout()
            
            # Prepare table data
            table_data = []
            for i, (feature, value, percent) in enumerate(zip(top_features, top_shap_values, percentages)):
                table_data.append({
                    'Rank': i + 1,
                    'Feature': feature,
                    'SHAP Value': f"{value:.6f}",
                    'Impact %': f"{percent:.2f}%",
                    'Effect': 'Supports Stroke' if value > 0 else 'Against Stroke'
                })
            
            return fig, table_data
            
        except Exception as e:
            st.error(f"❌ SHAP analysis failed: {str(e)}")
            return None, None

    def predict_image(self, processed_img):
        """REAL PREDICTION with WORKING SHAP analysis"""
        try:
            if not self.models_loaded:
                return None
                
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. Extract features
            status_text.text("🔍 Extracting features...")
            features = self.extract_features(processed_img)
            progress_bar.progress(30)
            
            if features is None:
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
            progress_bar.progress(85)
            
            # 5. Generate SHAP analysis
            shap_fig, shap_table_data = None, None
            status_text.text("📊 Generating feature analysis...")
            if SHAP_AVAILABLE:
                shap_fig, shap_table_data = self.generate_shap_analysis(selected_features)
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
            
            result = {
                'class': 'STROKE' if final_class == 1 else 'NORMAL',
                'confidence': float(confidence),
                'stroke_probability': stroke_prob,
                'normal_probability': normal_prob,
                'risk_level': risk_level,
                'emoji': emoji,
                'risk_class': risk_class,
                'shap_fig': shap_fig,
                'shap_table_data': shap_table_data,
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

def show_home_page():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to Brain Stroke Detection Pro
        
        Advanced AI system for brain stroke detection.
        
        ### 🔬 Features:
        - **Real Feature Extraction**: EfficientNetV2S, EfficientNetV2M, DenseNet201
        - **GWO Feature Selection**: Optimized feature selection
        - **Real Model Predictions**: Using your trained model
        - **Feature Importance Analysis**: Understand AI decisions
        
        ### 🚀 Get Started:
        1. Go to the **Analysis** tab  
        2. Upload a brain CT scan image
        3. View AI analysis with feature explanations
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2771/2771089.png", width=200)
        st.info("""
        **⚠️ Medical Disclaimer**
        For research purposes only. 
        Consult healthcare professionals.
        """)

def show_analysis_page():
    st.header("📊 Brain CT Analysis")
    
    if not TENSORFLOW_AVAILABLE or not JOBLIB_AVAILABLE:
        st.error("❌ Required dependencies not available")
        return
    
    # Load detector
    detector = load_detector()
    if detector is None:
        st.info("""
        💡 **If models fail to load:**
        - Ensure these files are present:
          - `final_ultimate_model_gwo.h5`
          - `final_ultimate_scaler_gwo.pkl`
          - `gwo_feature_mask.npy`
        """)
        return
    
    uploaded_file = st.file_uploader(
        "Upload Brain CT Image", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        # Create temporary file
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
                            
                            st.markdown(f"**AI Confidence:** {result['confidence']:.3f}")
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
                                'Probability': [
                                    result['normal_probability'], 
                                    result['stroke_probability']
                                ]
                            }
                            st.bar_chart(chart_data, x='Category', y='Probability')
                        
                        # Feature Importance Section
                        st.markdown("---")
                        st.subheader("🔍 Feature Importance Analysis")
                        
                        if result['shap_fig'] is not None and result['shap_table_data'] is not None:
                            # Display SHAP plot
                            st.markdown("### 📊 Top 10 Influential Features")
                            st.pyplot(result['shap_fig'])
                            st.caption("🟢 Green bars: Features supporting stroke prediction | 🔴 Red bars: Features against stroke prediction")
                            
                            # Display SHAP table
                            st.markdown("### 📋 Feature Impact Details")
                            st.table(result['shap_table_data'])
                            
                            # Interpretation
                            st.markdown("#### 💡 Interpretation:")
                            positive_features = [f for f in result['shap_table_data'] if f['Effect'] == 'Supports Stroke']
                            negative_features = [f for f in result['shap_table_data'] if f['Effect'] == 'Against Stroke']
                            
                            if positive_features:
                                st.write(f"✅ **{len(positive_features)} features** are supporting stroke prediction")
                            if negative_features:
                                st.write(f"❌ **{len(negative_features)} features** are against stroke prediction")
                                
                        else:
                            st.info("ℹ️ Feature importance analysis not available")
                        
                        # Final message
                        st.markdown("---")
                        st.markdown(f"### 💡 {result['message']}")
                    
                    else:
                        st.error("❌ Prediction failed.")
                else:
                    st.error("❌ Image processing failed.")
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        st.warning("**⚠️ Medical Disclaimer:** For research purposes only.")

def show_file_setup_page():
    st.header("🛠️ File Setup Guide")
    
    st.markdown("""
    ## Required Files:
    ```
    final_ultimate_model_gwo.h5
    final_ultimate_scaler_gwo.pkl  
    gwo_feature_mask.npy
    ```
    """)
    
    # Check files
    files_to_check = [
        "final_ultimate_model_gwo.h5",
        "final_ultimate_scaler_gwo.pkl",
        "gwo_feature_mask.npy"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 File Status")
        for file in files_to_check:
            if os.path.exists(file):
                st.success(f"✅ {file}")
            else:
                st.error(f"❌ {file}")
    
    with col2:
        st.subheader("🔗 Split Files")
        split_files = [f for f in os.listdir('.') if 'final_ultimate_model_gwo.h5.part' in f]
        if split_files:
            for file in sorted(split_files):
                st.info(f"📦 {file}")
        else:
            st.info("No split files found")

def main():
    # Sidebar status
    with st.sidebar:
        st.subheader("🔧 System Status")
        st.write(f"OpenCV: {'✅' if CV2_AVAILABLE else '❌'}")
        st.write(f"TensorFlow: {'✅' if TENSORFLOW_AVAILABLE else '❌'}")
        st.write(f"Joblib: {'✅' if JOBLIB_AVAILABLE else '❌'}")
        st.write(f"SHAP: {'✅' if SHAP_AVAILABLE else '❌'}")
        st.write(f"Files: {'✅' if combine_success else '❌'}")
    
    st.markdown('<h1 class="main-header">🧠 Brain Stroke Detection Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced AI-powered Stroke Detection with Feature Analysis")
    
    # Navigation
    app_mode = st.sidebar.selectbox(
        "Choose Mode", 
        ["🏠 Home", "📊 Analysis", "🛠️ File Setup"],
        key="nav_select"
    )
    
    if app_mode == "🏠 Home":
        show_home_page()
    elif app_mode == "📊 Analysis":
        show_analysis_page()
    else:
        show_file_setup_page()

if __name__ == "__main__":
    main()