import json
import os
import random

def debug_template_issue():
    print("Debugging Template Selection Issue")
    print("=" * 50)
    
    # Load metadata correctly
    metadata_file = "vitonhd_train_tagged.json"
    print(f"Loading metadata from: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    
    metadata = data['data']
    print(f"Total metadata items: {len(metadata)}")
    
    # Check cloth directory
    cloth_dir = "train/cloth"
    print(f"\nChecking cloth directory: {cloth_dir}")
    
    if os.path.exists(cloth_dir):
        cloth_files = os.listdir(cloth_dir)
        print(f"Total cloth files: {len(cloth_files)}")
        
        # Match metadata to actual files
        matched_items = []
        for item in metadata[:100]:  # Check first 100
            file_name = item.get('file_name', '')
            if file_name and file_name in cloth_files:
                matched_items.append(item)
        
        print(f"Matched metadata to files (first 100): {len(matched_items)}")
        
        # Test template selection logic
        categories = {}
        for item in matched_items:
            category = item.get('category_name', 'UNKNOWN')
            if category not in categories:
                categories[category] = []
            categories[category].append(item)
        
        print(f"Categories found: {list(categories.keys())}")
        for cat, items in categories.items():
            print(f"  {cat}: {len(items)} items")
        
        # Select templates
        templates = []
        images_per_category = max(1, 30 // len(categories)) if categories else 30
        
        for category, items in categories.items():
            # Random sample from each category
            sample_size = min(images_per_category, len(items))
            selected = random.sample(items, sample_size)
            templates.extend(selected)
        
        print(f"\nSelected templates: {len(templates)}")
        
        # Show sample templates
        if templates:
            print("Sample selected templates:")
            for i, template in enumerate(templates[:5]):
                print(f"  {i+1}. {template['file_name']} - {template.get('category_name', 'UNKNOWN')}")
    else:
        print(f"❌ Cloth directory not found: {cloth_dir}")

if __name__ == "__main__":
    debug_template_issue()
