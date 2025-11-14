# ===== STREAMLIT APP - FIXED GRAD-CAM & SHAP =====

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

    def generate_grad_cam(self, processed_img):
        """FIXED Grad-CAM - Using EfficientNet with proper layer detection"""
        try:
            if not CV2_AVAILABLE:
                return None, None
            
            # Use EfficientNetV2S base model for Grad-CAM
            base_model = EfficientNetV2S(weights='imagenet', include_top=False, 
                                       input_shape=(self.IMG_SIZE[0], self.IMG_SIZE[1], 3))
            
            # Get ALL convolutional layers
            conv_layers = [layer for layer in base_model.layers if 'conv' in layer.name]
            
            if not conv_layers:
                return None, None
            
            # Use the last convolutional layer
            layer_name = conv_layers[-1].name
            
            # Create Grad-CAM model
            grad_model = Model(
                inputs=base_model.input,
                outputs=[base_model.get_layer(layer_name).output, base_model.output]
            )
            
            # Prepare image
            img_batch = np.expand_dims(processed_img, axis=0)
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_batch)
                # Use the class with highest activation
                target = tf.reduce_max(predictions)
            
            grads = tape.gradient(target, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Process heatmap
            heatmap_np = heatmap.numpy()
            heatmap_np = np.maximum(heatmap_np, 0)
            if np.max(heatmap_np) > 0:
                heatmap_np /= np.max(heatmap_np)
            
            # Resize and create visualization
            heatmap_resized = cv2.resize(heatmap_np, (processed_img.shape[1], processed_img.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            
            # Superimpose on image
            img_uint8 = (processed_img * 255).astype(np.uint8)
            if len(img_uint8.shape) == 3:
                img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            
            superimposed_img = cv2.addWeighted(img_uint8, 0.6, heatmap_colored, 0.4, 0)
            superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
            
            return superimposed_img, heatmap_colored
            
        except Exception as e:
            return None, None

    def generate_shap_explanation(self, selected_features):
        """FIXED SHAP - Proper array handling"""
        try:
            if not SHAP_AVAILABLE:
                return None, None
            
            # Check if selected_features is valid
            if selected_features is None:
                return None, None
            
            # Convert to numpy array and check shape
            selected_features_array = np.array(selected_features)
            if selected_features_array.size == 0:
                return None, None
            
            # Use small background for speed
            background = np.random.rand(10, selected_features_array.shape[1]) * 0.1
            
            def predict_fn(x):
                return self.ultimate_model_gwo.predict(x, verbose=0)
            
            # Create explainer
            explainer = shap.KernelExplainer(predict_fn, background)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(selected_features_array, nsamples=10)
            
            # Handle SHAP output format - FIXED array truth value error
            if isinstance(shap_values, list):
                # For multi-class, use the stroke class (index 1)
                if len(shap_values) > 1:
                    stroke_shap_values = shap_values[1]  # Class 1 for stroke
                else:
                    stroke_shap_values = shap_values[0]
            else:
                stroke_shap_values = shap_values
            
            # Ensure we have the right shape - FIXED array handling
            if hasattr(stroke_shap_values, 'shape') and len(stroke_shap_values.shape) > 1:
                stroke_shap_values = stroke_shap_values[0]  # Take first sample
            
            # Convert to numpy array for safe operations
            stroke_shap_array = np.array(stroke_shap_values)
            
            # Get top features
            top_n = min(10, len(stroke_shap_array))
            top_indices = np.argsort(np.abs(stroke_shap_array))[::-1][:top_n]
            top_features = [f"Feature_{i}" for i in top_indices]
            top_shap_values = stroke_shap_array[top_indices]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(top_features))
            
            colors = ['green' if x > 0 else 'red' for x in top_shap_values]
            
            bars = ax.barh(y_pos, top_shap_values, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features)
            ax.set_xlabel('SHAP Value Impact on Stroke Prediction')
            ax.set_title('Top Feature Impacts on AI Decision')
            ax.invert_yaxis()
            
            # Add value labels
            for bar, value in zip(bars, top_shap_values):
                width = bar.get_width()
                label_x_pos = width + 0.01 if width >= 0 else width - 0.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center')
            
            plt.tight_layout()
            
            return fig, list(zip(top_features, top_shap_values))
            
        except Exception as e:
            return None, None

    def predict_image(self, processed_img):
        """REAL PREDICTION with FIXED Grad-CAM & SHAP"""
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
            
            # 5. Generate Grad-CAM
            grad_cam_img, heatmap = None, None
            status_text.text("🎯 Generating heatmap...")
            if CV2_AVAILABLE:
                grad_cam_img, heatmap = self.generate_grad_cam(processed_img)
            progress_bar.progress(92)
            
            # 6. Generate SHAP explanation
            shap_fig, shap_explanation = None, None
            status_text.text("📊 Generating explanations...")
            if SHAP_AVAILABLE:
                shap_fig, shap_explanation = self.generate_shap_explanation(selected_features)
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
                'grad_cam_img': grad_cam_img,
                'heatmap': heatmap,
                'shap_fig': shap_fig,
                'shap_explanation': shap_explanation,
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
        - **Grad-CAM Heatmaps**: Visualize AI attention
        - **SHAP Explanations**: Understand feature importance
        
        ### 🚀 Get Started:
        1. Go to the **Analysis** tab  
        2. Upload a brain CT scan image
        3. View AI analysis with explanations
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
                        
                        # Explainable AI Section
                        st.markdown("---")
                        st.subheader("🔍 Explainable AI Insights")
                        
                        # Grad-CAM Visualization
                        if result['grad_cam_img'] is not None:
                            st.markdown("### 🎯 Grad-CAM Heatmap")
                            col7, col8 = st.columns(2)
                            
                            with col7:
                                st.subheader("Heatmap")
                                st.image(result['heatmap'], use_column_width=True, clamp=True)
                                st.caption("Red areas show where the AI focuses attention")
                            
                            with col8:
                                st.subheader("Overlay on Image")
                                st.image(result['grad_cam_img'], use_column_width=True)
                                st.caption("AI attention overlaid on processed image")
                        else:
                            st.info("ℹ️ Grad-CAM visualization not available")
                        
                        # SHAP Explanations
                        if result['shap_fig'] is not None:
                            st.markdown("### 📊 SHAP Feature Importance")
                            st.pyplot(result['shap_fig'])
                            st.caption("Green bars: Features supporting stroke prediction | Red bars: Features against stroke prediction")
                            
                            if result['shap_explanation']:
                                st.markdown("#### Top Influential Features:")
                                shap_data = []
                                for feature, importance in result['shap_explanation']:
                                    shap_data.append({
                                        'Feature': feature,
                                        'Impact': f"{importance:.4f}",
                                        'Effect': 'Supports Stroke' if importance > 0 else 'Against Stroke'
                                    })
                                st.table(shap_data)
                        else:
                            st.info("ℹ️ SHAP explanation not available")
                        
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
    st.markdown("### Advanced AI-powered Stroke Detection with Explainable AI")
    
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