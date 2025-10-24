
import os
import streamlit as st

def verify_files():
    st.title("🔍 File Verification")
    
    # Check for all required files
    required_files = [
        'app.py',
        'requirements.txt',
        'runtime.txt', 
        'best_bilstm.keras',
        'scaler_features.pkl',
        'best_mask.npy'
    ]
    
    # Check for part files
    part_files = [
        'feature_extractor.keras.part000',
        'feature_extractor.keras.part001',
        'feature_extractor.keras.part002',
        'feature_extractor.keras.part003',
        'feature_extractor.keras.part004',
        'feature_extractor.keras.part005',
        'feature_extractor.keras.part006'
    ]
    
    st.subheader("Required Files:")
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)
            st.success(f"✅ {file} - {size:.2f} MB")
        else:
            st.error(f"❌ {file} - MISSING")
    
    st.subheader("Part Files:")
    for file in part_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)
            st.success(f"✅ {file} - {size:.2f} MB")
        else:
            st.error(f"❌ {file} - MISSING")
    
    st.subheader("All Files in Directory:")
    for file in sorted(os.listdir('.')):
        size = os.path.getsize(file) / (1024 * 1024)
        st.write(f"📄 {file} - {size:.2f} MB")

if __name__ == "__main__":
    verify_files()
