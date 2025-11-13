import os
import sys

def combine_model_chunks():
    """Combine model chunk files if they exist"""
    part_files = [f for f in os.listdir('.') if f.endswith('.part')]
    if not part_files:
        print("No model chunks found to combine")
        return False
    
    # Sort part files to combine in correct order
    part_files.sort()
    
    try:
        with open('final_ultimate_model_gwo.h5', 'wb') as outfile:
            for part_file in part_files:
                with open(part_file, 'rb') as infile:
                    outfile.write(infile.read())
        print(f"Successfully combined {len(part_files)} chunks into final_ultimate_model_gwo.h5")
        return True
    except Exception as e:
        print(f"Error combining chunks: {e}")
        return False

if __name__ == "__main__":
    combine_model_chunks()
