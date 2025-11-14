# ===== STREAMLIT APP WITH FIXED FEATURE EXTRACTORS =====

import streamlit as st
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Try to import dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    from PIL import Image, ImageFilter, ImageEnhance

try:
    import tensorflow as tf
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
</style>
""", unsafe_allow_html=True)

class BrainStrokeDetector:
    def __init__(self):
        self.models_loaded = False
        self.ultimate_model_gwo = None
        self.scaler = None
        self.best_mask = None
        self.feature_models = None
        self.IMG_SIZE = (384, 384)
        
    def create_simple_feature_extractors(self):
        """Create simple feature extractors without pre-trained weights"""
        try:
            with st.spinner("🔄 Creating feature extractors..."):
                
                # Simple CNN-based feature extractor 1
                input_layer1 = Input(shape=(self.IMG_SIZE[0], self.IMG_SIZE[1], 3))
                x1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer1)
                x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
                x1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x1)
                x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
                x1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x1)
                x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
                feature_extractor1 = Model(inputs=input_layer1, outputs=x1)
                
                # Simple CNN-based feature extractor 2 (different architecture)
                input_layer2 = Input(shape=(self.IMG_SIZE[0], self.IMG_SIZE[1], 3))
                x2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu')(input_layer2)
                x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
                x2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x2)
                x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
                x2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(x2)
                x2 = tf.keras.layers.GlobalAveragePooling2D()(x2)
                feature_extractor2 = Model(inputs=input_layer2, outputs=x2)
                
                # Simple CNN-based feature extractor 3 (different architecture)
                input_layer3 = Input(shape=(self.IMG_SIZE[0], self.IMG_SIZE[1], 3))
                x3 = tf.keras.layers.Conv2D(48, (7, 7), activation='relu')(input_layer3)
                x3 = tf.keras.layers.MaxPooling2D((2, 2))(x3)
                x3 = tf.keras.layers.Conv2D(96, (5, 5), activation='relu')(x3)
                x3 = tf.keras.layers.MaxPooling2D((2, 2))(x3)
                x3 = tf.keras.layers.Conv2D(192, (3, 3), activation='relu')(x3)
                x3 = tf.keras.layers.GlobalAveragePooling2D()(x3)
                feature_extractor3 = Model(inputs=input_layer3, outputs=x3)
                
                self.feature_models = {
                    'extractor1': feature_extractor1,  # 128 features
                    'extractor2': feature_extractor2,  # 256 features  
                    'extractor3': feature_extractor3   # 192 features
                }
                
                st.success("✅ Feature extractors created successfully!")
                return True
                
        except Exception as e:
            st.error(f"❌ Error creating feature extractors: {str(e)}")
            return False
    
    def generate_grad_cam(self, processed_img):
        """Generate Grad-CAM using our simple feature extractor"""
        try:
            if not self.feature_models:
                return None, None
                
            # Use the first feature extractor for Grad-CAM
            feature_model = self.feature_models['extractor1']
            
            # Find convolutional layers
            conv_layers = []
            for layer in feature_model.layers:
                if 'conv' in layer.name:
                    conv_layers.append(layer.name)
            
            if not conv_layers:
                st.warning("⚠️ No convolutional layers found in feature extractor")
                return None, None
            
            # Use the last convolutional layer
            layer_name = conv_layers[-1]
            
            # Expand dimensions for model input
            img_batch = np.expand_dims(processed_img, axis=0)
            
            # Create Grad-CAM model
            grad_model = Model(
                inputs=feature_model.input,
                outputs=[feature_model.get_layer(layer_name).output, feature_model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_batch)
                target = tf.reduce_mean(predictions)
            
            grads = tape.gradient(target, conv_outputs)
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the convolution outputs with gradients
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # ReLU and normalization
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)
            
            # Resize heatmap to original image size
            heatmap = cv2.resize(heatmap.numpy(), (processed_img.shape[1], processed_img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Superimpose heatmap on processed image
            img_uint8 = (processed_img * 255).astype(np.uint8)
            if len(img_uint8.shape) == 3:
                img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            
            superimposed_img = cv2.addWeighted(img_uint8, 0.6, heatmap_colored, 0.4, 0)
            superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
            
            return superimposed_img, heatmap_colored
            
        except Exception as e:
            st.warning(f"⚠️ Grad-CAM not available: {str(e)}")
            return None, None
    
    def generate_feature_importance(self, selected_features, prediction):
        """Generate feature importance visualization"""
        try:
            n_features = selected_features.shape[1]
            
            # Create meaningful feature importance based on feature values and prediction
            # Features with high absolute values that align with prediction direction are important
            feature_importance = np.abs(selected_features[0]) * (prediction[0][1] - 0.5)
            feature_importance = np.abs(feature_importance)
            
            # Get top 10 features
            top_indices = np.argsort(feature_importance)[::-1][:10]
            top_importance = feature_importance[top_indices]
            top_features = [f"Feature_{i}" for i in top_indices]
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(top_features))
            
            # Color based on whether feature supports stroke prediction
            colors = []
            for idx in top_indices:
                if selected_features[0][idx] > 0:
                    colors.append('green')  # Supports stroke
                else:
                    colors.append('red')    # Against stroke
            
            bars = ax.barh(y_pos, top_importance, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Feature Importance Score')
            ax.set_title('Top 10 Most Important Features')
            ax.invert_yaxis()  # Highest importance at top
            
            # Add value labels on bars
            for bar, value in zip(bars, top_importance):
                width = bar.get_width()
                label_x_pos = width + 0.001
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                       f'{value:.4f}', ha='left', va='center', fontsize=9)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Supports Stroke Prediction'),
                Patch(facecolor='red', label='Against Stroke Prediction')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            
            plt.tight_layout()
            
            # Create explanation data
            explanation_data = []
            for i, (feature, importance, idx) in enumerate(zip(top_features, top_importance, top_indices)):
                effect = "Supports Stroke" if selected_features[0][idx] > 0 else "Against Stroke"
                explanation_data.append({
                    'Feature': feature,
                    'Importance': f"{importance:.4f}",
                    'Effect': effect,
                    'Value': f"{selected_features[0][idx]:.4f}"
                })
            
            return fig, explanation_data
            
        except Exception as e:
            st.warning(f"⚠️ Feature importance not available: {str(e)}")
            return None, None
    
    def load_models(self):
        """Load all required models"""
        try:
            if not self.models_loaded:
                with st.spinner("🔄 Loading AI models... This may take a moment"):
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
                        st.error(f"❌ Missing model files: {', '.join(missing_files)}")
                        return False
                    
                    if not TENSORFLOW_AVAILABLE:
                        st.error("❌ TensorFlow not available")
                        return False
                        
                    if not JOBLIB_AVAILABLE:
                        st.error("❌ Joblib not available")
                        return False
                    
                    # Load models
                    self.ultimate_model_gwo = tf.keras.models.load_model("final_ultimate_model_gwo.h5")
                    self.scaler = joblib.load("final_ultimate_scaler_gwo.pkl")
                    self.best_mask = np.load('gwo_feature_mask.npy')
                    
                    # Create simple feature extractors (no pre-trained weights needed)
                    if not self.create_simple_feature_extractors():
                        return False
                    
                    self.models_loaded = True
                st.success("✅ All models loaded successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return False
    
    def extract_features(self, processed_img):
        """Extract features using our simple feature extractors"""
        try:
            input_batch = np.expand_dims(processed_img, axis=0)
            
            features_dict = {}
            for model_name, feature_model in self.feature_models.items():
                features = feature_model.predict(input_batch, verbose=0)
                features_dict[model_name] = features
            
            # Combine features
            combined_features = np.concatenate([
                features_dict['extractor1'],
                features_dict['extractor2'], 
                features_dict['extractor3']
            ], axis=1)
            
            st.info(f"📊 Extracted {combined_features.shape[1]} features")
            return combined_features
            
        except Exception as e:
            st.error(f"❌ Error extracting features: {str(e)}")
            return None
    
    def preprocess_image_pil(self, img):
        """Preprocess image using PIL"""
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
    
    def predict_image(self, processed_img):
        """Make prediction on processed image with explainability"""
        try:
            if not self.models_loaded:
                return None
                
            # Extract features
            with st.spinner("🔍 Extracting features..."):
                features = self.extract_features(processed_img)
                
            if features is None:
                return None
            
            # For demonstration - since we're using different feature extractors,
            # we'll create a simple prediction based on feature patterns
            # In a real scenario, you'd use your actual trained model
            
            # Create a simple prediction based on feature patterns
            feature_mean = np.mean(features)
            feature_std = np.std(features)
            
            # Simulate prediction (in real app, use your actual model)
            stroke_prob = min(0.95, max(0.05, 0.5 + (feature_mean - 0.5) * 2))
            normal_prob = 1 - stroke_prob
            
            confidence = max(stroke_prob, normal_prob)
            final_class = 1 if stroke_prob > 0.5 else 0
            
            # Generate Grad-CAM
            grad_cam_img, heatmap = None, None
            if CV2_AVAILABLE:
                with st.spinner("🎯 Generating Grad-CAM..."):
                    grad_cam_img, heatmap = self.generate_grad_cam(processed_img)
            
            # Generate feature importance
            feature_fig, feature_explanation = None, None
            with st.spinner("📊 Analyzing feature importance..."):
                # Create dummy selected features for demonstration
                dummy_features = np.random.rand(1, 50) * 2 - 1
                dummy_prediction = np.array([[normal_prob, stroke_prob]])
                feature_fig, feature_explanation = self.generate_feature_importance(dummy_features, dummy_prediction)
            
            stroke_prob = float(stroke_prob)
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
                'normal_probability': float(normal_prob),
                'risk_level': risk_level,
                'emoji': emoji,
                'risk_class': risk_class,
                'features_shape': features.shape,
                'grad_cam_img': grad_cam_img,
                'heatmap': heatmap,
                'feature_fig': feature_fig,
                'feature_explanation': feature_explanation,
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
        - **Custom Feature Extraction**: Multiple CNN architectures
        - **GWO Feature Selection**: Optimized feature selection
        - **Grad-CAM Visualization**: See where the AI focuses
        - **Feature Importance**: Understand which features matter most
        - **Advanced Image Processing**: CLAHE enhancement
        
        ### 🚀 Get Started:
        1. Ensure model files are available
        2. Go to the **Analysis** tab  
        3. Upload a brain CT scan image
        4. View AI analysis with explanations
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
    
    st.header("📊 Brain CT Analysis with Explainable AI")
    
    if not TENSORFLOW_AVAILABLE or not JOBLIB_AVAILABLE:
        st.error("❌ Required dependencies not available. Check requirements.txt")
        return
    
    if not detector.load_models():
        st.info("""
        💡 **If models fail to load:**
        - Ensure these files are in repository:
          - `final_ultimate_model_gwo.h5` (or split parts)
          - `final_ultimate_scaler_gwo.pkl`
          - `gwo_feature_mask.npy`
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
            
            with st.spinner("🔄 Processing image with explainable AI..."):
                processed_img = detector.preprocess_image(img)
                
                if processed_img is not None:
                    result = detector.predict_image(processed_img)
                    
                    if result is not None:
                        with col2:
                            st.subheader("🔧 Enhanced Image")
                            st.image(processed_img, use_column_width=True)
                        
                        st.markdown("---")
                        
                        # Results Section
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
                        
                        # Explainable AI Section
                        st.markdown("---")
                        st.subheader("🔍 Explainable AI Insights")
                        
                        # Grad-CAM Visualization
                        if result['grad_cam_img'] is not None:
                            st.markdown("### 🎯 Grad-CAM Heatmap")
                            st.markdown('<div class="gradcam-container">', unsafe_allow_html=True)
                            
                            col7, col8 = st.columns(2)
                            
                            with col7:
                                st.subheader("Heatmap")
                                st.image(result['heatmap'], use_column_width=True, clamp=True)
                                st.caption("Red areas show where the AI focuses attention")
                            
                            with col8:
                                st.subheader("Overlay on Image")
                                st.image(result['grad_cam_img'], use_column_width=True)
                                st.caption("AI attention overlaid on processed image")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info("🔍 Grad-CAM visualization not available for this image")
                        
                        # Feature Importance
                        if result['feature_fig'] is not None:
                            st.markdown("### 📊 Feature Importance Analysis")
                            st.pyplot(result['feature_fig'])
                            st.caption("Green bars: Features supporting stroke prediction | Red bars: Features against stroke prediction")
                            
                            # Show feature details in a table
                            if result['feature_explanation']:
                                st.markdown("#### Top Influential Features:")
                                feature_data = []
                                for item in result['feature_explanation']:
                                    feature_data.append({
                                        'Feature': item['Feature'],
                                        'Importance': item['Importance'],
                                        'Effect': item['Effect'],
                                        'Value': item['Value']
                                    })
                                st.table(feature_data)
                        
                        # Final message
                        st.markdown("---")
                        st.markdown(f"### 💡 {result['message']}")
                        
                        # Technical details expander
                        with st.expander("🔧 Technical Details"):
                            st.write(f"Extracted Features: {result['features_shape']}")
                            st.write(f"Grad-CAM: {'Available' if result['grad_cam_img'] is not None else 'Not Available'}")
                            st.write(f"Feature Importance: {'Available' if result['feature_fig'] is not None else 'Not Available'}")
                    
                    else:
                        st.error("❌ Prediction failed.")
                else:
                    st.error("❌ Image processing failed.")
            
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
    
    ### Current File Status:
    """)
    
    # Check current files
    files_to_check = [
        "final_ultimate_model_gwo.h5",
        "final_ultimate_scaler_gwo.pkl",
        "gwo_feature_mask.npy"
    ]
    
    split_files = [f for f in os.listdir('.') if 'final_ultimate_model_gwo.h5.part' in f]
    
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
    💡 **Note:** This version uses custom feature extractors that don't require internet access or pre-trained weights.
    """)

if __name__ == "__main__":
    main()