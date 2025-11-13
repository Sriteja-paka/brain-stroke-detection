#!/bin/bash

echo "🚀 Setting up Brain Stroke Detection Environment..."

# Create virtual environment
python -m venv stroke_env
source stroke_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Combine model files if chunks exist
if ls final_ultimate_model_gwo.h5.part* 1> /dev/null 2>&1; then
    echo "🔗 Combining model file chunks..."
    python combine_model.py
fi

echo "✅ Setup complete!"
echo "🎯 To run the application:"
echo "   source stroke_env/bin/activate"
echo "   streamlit run app.py"