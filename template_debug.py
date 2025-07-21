import os
import random
import json

def debug_template_selection():
    print("Debug Template Selection")
    print("=" * 50)
    
    # Check cloth directory
    cloth_dir = "train/cloth"
    print(f"Checking directory: {cloth_dir}")
    print(f"Directory exists: {os.path.exists(cloth_dir)}")
    
    if os.path.exists(cloth_dir):
        files = os.listdir(cloth_dir)
        print(f"Total files in directory: {len(files)}")
        
        # Check for image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]
        print(f"Image files found: {len(image_files)}")
        
        if image_files:
            print("First 10 image files:")
            for i, img in enumerate(image_files[:10]):
                print(f"  {i+1}. {img}")
        
        # Test random selection
        if image_files:
            selected = random.sample(image_files, min(5, len(image_files)))
            print(f"\nRandom selection of {len(selected)} images:")
            for img in selected:
                print(f"  - {img}")
    
    # Check metadata
    print("\nChecking metadata...")
    metadata_file = "vitonhd_train_tagged.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"Metadata entries: {len(metadata)}")
        
        # Check cloth fields
        cloth_names = set()
        for item in metadata[:100]:  # Check first 100 items
            if 'cloth' in item:
                cloth_names.add(item['cloth'])
        
        print(f"Unique cloth names in first 100 entries: {len(cloth_names)}")
        if cloth_names:
            print("Sample cloth names:")
            for i, name in enumerate(list(cloth_names)[:5]):
                print(f"  - {name}")

if __name__ == "__main__":
    debug_template_selection()
