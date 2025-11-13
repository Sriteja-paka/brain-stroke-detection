# 🚀 Deployment Guide

## 1. Local Deployment
```bash
# Clone or download all files to a directory
cd brain-stroke-detection

# Run setup (Linux/Mac)
chmod +x setup.sh
./setup.sh

# Or manual setup
python -m venv stroke_env
source stroke_env/bin/activate
pip install -r requirements.txt

# Run application
streamlit run app.py
```

## 2. Cloud Deployment (Streamlit Cloud)

### Step 1: Upload to GitHub
1. Create new repository: `brain-stroke-detection`
2. Upload all files:
   - app.py
   - requirements.txt
   - README.md
   - setup.sh
   - combine_model.py
   - final_ultimate_scaler_gwo.pkl
   - gwo_feature_mask.npy
   - final_ultimate_model_gwo.h5.part* (all chunk files)

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select repository: `brain-stroke-detection`
4. Set main file path: `app.py`
5. Click "Deploy"

## 3. Required Files
Ensure these files are in the same directory:
- `app.py` - Main application
- `requirements.txt` - Dependencies
- `final_ultimate_scaler_gwo.pkl` - Feature scaler
- `gwo_feature_mask.npy` - Feature mask
- `final_ultimate_model_gwo.h5` OR `final_ultimate_model_gwo.h5.part*` - Model

## 4. Model File Strategy
- **If model <25MB**: Upload `final_ultimate_model_gwo.h5` directly
- **If model >25MB**: Split into chunks using the provided script

## 📊 Performance Metrics
- Model Loading: ~30 seconds
- Prediction Time: ~5 seconds
- Accuracy: 97%+
- Memory: ~2GB RAM required