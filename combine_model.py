#!/usr/bin/env python3
"""
Brain Stroke Detection - Robust Model File Combiner
Automatically combines model chunks and verifies the result
"""

import os
import glob
import hashlib
import sys

def get_file_hash(filename):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        return None

def combine_model_files():
    print("🧠 Brain Stroke Detection - Model File Combiner")
    print("=" * 50)
    
    # Look for different possible model file patterns
    possible_patterns = [
        "final_ultimate_model_gwo.h5.part*",
        "best_ultimate_model_gwo.h5.part*", 
        "*.h5.part*"
    ]
    
    part_files = []
    for pattern in possible_patterns:
        files = glob.glob(pattern)
        if files:
            part_files.extend(files)
    
    # Remove duplicates and sort
    part_files = sorted(list(set(part_files)))
    
    if not part_files:
        print("❌ No model chunk files found!")
        print("   Looking for files like: final_ultimate_model_gwo.h5.part*")
        print("   Available files in directory:")
        for f in os.listdir('.'):
            if '.part' in f or '.h5' in f or '.pkl' in f or '.npy' in f:
                print(f"     - {f}")
        return False
    
    print(f"📁 Found {len(part_files)} model chunks:")
    for f in part_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"   🔍 {f} ({size_mb:.1f} MB)")
    
    # Determine output filename from first part file
    first_part = part_files[0]
    if 'final_ultimate' in first_part:
        output_file = 'final_ultimate_model_gwo.h5'
    elif 'best_ultimate' in first_part:
        output_file = 'best_ultimate_model_gwo.h5'
    else:
        # Extract base name (remove .part*)
        base_name = first_part.split('.part')[0]
        output_file = base_name
    
    print(f"🎯 Output file: {output_file}")
    
    # Check if output already exists
    if os.path.exists(output_file):
        existing_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"⚠️  {output_file} already exists ({existing_size:.1f} MB)")
        response = input("   Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("❌ Operation cancelled")
            return False
    
    # Combine files
    try:
        total_size = 0
        with open(output_file, 'wb') as outfile:
            for i, part_file in enumerate(part_files, 1):
                print(f"🔗 Combining part {i}/{len(part_files)}: {part_file}")
                with open(part_file, 'rb') as infile:
                    data = infile.read()
                    outfile.write(data)
                    total_size += len(data)
        
        # Verify the combined file
        if os.path.exists(output_file):
            final_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"✅ Successfully created {output_file}")
            print(f"📊 Final size: {final_size:.1f} MB")
            print(f"📦 Total chunks processed: {len(part_files)}")
            
            # Verify file integrity
            print("🔍 Verifying file integrity...")
            if final_size > 0:
                print("🎉 Model file ready for use!")
                print("🚀 You can now run: streamlit run app.py")
                return True
            else:
                print("❌ Combined file is empty!")
                return False
        else:
            print("❌ Failed to create output file")
            return False
            
    except Exception as e:
        print(f"❌ Error combining files: {e}")
        # Clean up partial output
        if os.path.exists(output_file):
            os.remove(output_file)
        return False

if __name__ == "__main__":
    success = combine_model_files()
    sys.exit(0 if success else 1)
