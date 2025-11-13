#!/usr/bin/env python3
"""
Brain Stroke Detection - Model File Combiner
Run this script to reassemble the model file from chunks
"""

import os
import glob

def combine_model_files():
    print("🧠 Brain Stroke Detection - Model File Combiner")
    print("=" * 50)
    
    # Find all part files
    part_files = glob.glob("final_ultimate_model_gwo.h5.part*")
    part_files.sort()
    
    if not part_files:
        print("❌ No part files found!")
        print("   Make sure all .part files are in the same directory")
        return False
    
    print(f"📁 Found {len(part_files)} part files")
    
    # Combine files
    output_file = "final_ultimate_model_gwo.h5"
    try:
        with open(output_file, "wb") as outfile:
            for part_file in part_files:
                print(f"🔗 Adding {part_file}...")
                with open(part_file, "rb") as infile:
                    outfile.write(infile.read())
        
        # Verify file size
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"✅ Successfully created {output_file} ({file_size:.1f} MB)")
            print("🎯 You can now run: streamlit run app.py")
            return True
        else:
            print("❌ Failed to create output file")
            return False
            
    except Exception as e:
        print(f"❌ Error combining files: {e}")
        return False

if __name__ == "__main__":
    combine_model_files()