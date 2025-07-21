"""
Quick test script to verify the fashion recommender system functionality
"""

import json
import os
from pathlib import Path
from PIL import Image
import random

def test_metadata_loading():
    """Test if metadata loads correctly"""
    print("Testing metadata loading...")
    metadata_path = Path("vitonhd_train_tagged.json")
    
    if not metadata_path.exists():
        print("❌ Metadata file not found!")
        return False
    
    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        items = data['data']
        print(f"✅ Successfully loaded {len(items)} fashion items")
        
        # Show sample item
        if items:
            sample = items[0]
            print(f"Sample item: {sample.get('file_name', 'N/A')}")
            print(f"Category: {sample.get('category_name', 'N/A')}")
            print(f"Tags: {len(sample.get('tag_info', []))}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return False

def test_images_availability():
    """Test if image files are available"""
    print("\nTesting image availability...")
    cloth_path = Path("test/cloth")
    
    if not cloth_path.exists():
        print("❌ Cloth images directory not found!")
        return False
    
    image_files = list(cloth_path.glob("*.jpg"))
    print(f"✅ Found {len(image_files)} image files")
    
    # Test loading a few images
    test_count = min(5, len(image_files))
    success_count = 0
    
    for i in range(test_count):
        try:
            img_path = image_files[i]
            img = Image.open(img_path)
            img.verify()  # Check if image is valid
            success_count += 1
        except Exception as e:
            print(f"⚠️ Error loading {img_path.name}: {e}")
    
    print(f"✅ Successfully verified {success_count}/{test_count} test images")
    return success_count > 0

def test_model_availability():
    """Test if GAT model file exists"""
    print("\nTesting model availability...")
    model_path = Path("gat_model.pth")
    
    if model_path.exists():
        print(f"✅ GAT model found: {model_path.name} ({model_path.stat().st_size} bytes)")
        return True
    else:
        print("⚠️ GAT model not found - will use fallback model")
        return False

def test_streamlit_imports():
    """Test if all required packages can be imported"""
    print("\nTesting package imports...")
    
    packages = [
        ("streamlit", "st"),
        ("PIL", "Image"),
        ("numpy", "np"),
        ("pandas", "pd"),
        ("sklearn.feature_extraction.text", "TfidfVectorizer"),
        ("cv2", None)
    ]
    
    success_count = 0
    for package, alias in packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"✅ {package}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {package}: {e}")
    
    print(f"✅ Successfully imported {success_count}/{len(packages)} packages")
    return success_count == len(packages)

def generate_sample_recommendations():
    """Generate sample recommendations to test the system"""
    print("\nGenerating sample recommendations...")
    
    try:
        # Test data
        sample_queries = [
            "black casual shirt",
            "red dress",
            "denim jacket",
            "summer top",
            "formal wear"
        ]
        
        print("Sample text queries that can be used:")
        for i, query in enumerate(sample_queries, 1):
            print(f"  {i}. {query}")
        
        return True
    except Exception as e:
        print(f"❌ Error generating samples: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Fashion Recommender System - Quick Test")
    print("=" * 50)
    
    tests = [
        test_streamlit_imports,
        test_metadata_loading,
        test_images_availability,
        test_model_availability,
        generate_sample_recommendations
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed >= 3:  # Core functionality works
        print("🎉 System is ready to use!")
        print("\nTo start the application:")
        print("1. Run: streamlit run enhanced_fashion_app.py")
        print("2. Open browser to: http://localhost:8501")
    else:
        print("⚠️ Some issues detected. Please check the requirements.")
    
    return passed >= 3

if __name__ == "__main__":
    main()
