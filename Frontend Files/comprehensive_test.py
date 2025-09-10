"""
Comprehensive test for the enhanced fashion recommendation system
Tests both the correct GAT architecture and Streamlit integration
"""

import os
import json
import sys
from pathlib import Path

def test_gat_architecture():
    """Test the GAT architecture loading"""
    print("🧪 Testing GAT Architecture...")
    
    try:
        from model1_architecture import FashionGATModel
        
        # Test model loading
        model = FashionGATModel("gat_model.pth")
        print(f"✅ GAT Model loaded: {model.model_loaded}")
        
        # Test with sample metadata
        sample_metadata = [
            {
                "file_name": "test.jpg",
                "category_name": "TOPS",
                "tag_info": [
                    {"tag_name": "item", "tag_category": "T-Shirts"},
                    {"tag_name": "colors", "tag_category": "Blue"}
                ]
            }
        ] * 5
        
        features = model.extract_features(sample_metadata)
        print(f"✅ Feature extraction successful! Shape: {features.shape}")
        
        similarities = model.find_similar_items(0, features, top_k=3)
        print(f"✅ Similarity computation successful! Found {len(similarities)} similar items")
        
        return True
        
    except Exception as e:
        print(f"❌ GAT architecture test failed: {e}")
        return False

def test_enhanced_integration():
    """Test the enhanced GAT integration"""
    print("🧪 Testing Enhanced GAT Integration...")
    
    try:
        from enhanced_gat_integration import EnhancedFashionGATRecommender
        
        # Test recommender initialization
        recommender = EnhancedFashionGATRecommender("gat_model.pth")
        print(f"✅ Enhanced recommender initialized: {recommender.model_loaded}")
        
        # Test metadata recommendation
        sample_metadata = [
            {
                "file_name": "test1.jpg",
                "category_name": "TOPS",
                "tag_info": [
                    {"tag_name": "item", "tag_category": "T-Shirts"},
                    {"tag_name": "colors", "tag_category": "Blue"}
                ]
            },
            {
                "file_name": "test2.jpg",
                "category_name": "TOPS", 
                "tag_info": [
                    {"tag_name": "item", "tag_category": "Polo Shirts"},
                    {"tag_name": "colors", "tag_category": "Red"}
                ]
            }
        ] * 3
        
        similarities = recommender.find_similar_by_index(0, sample_metadata, top_k=2)
        print(f"✅ Metadata recommendation successful! Found {len(similarities)} similar items")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced integration test failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset and metadata loading"""
    print("🧪 Testing Dataset Loading...")
    
    try:
        # Check if train dataset exists
        train_cloth_path = Path("train/cloth")
        if train_cloth_path.exists():
            cloth_files = list(train_cloth_path.glob("*.jpg"))
            print(f"✅ Found {len(cloth_files)} images in train/cloth directory")
        else:
            print("⚠️ train/cloth directory not found")
        
        # Check metadata
        metadata_path = Path("vitonhd_train_tagged.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Found {len(metadata['data'])} metadata items")
            
            # Check a sample item
            sample_item = metadata['data'][0]
            print(f"✅ Sample item structure: {list(sample_item.keys())}")
            
            return True
        else:
            print("❌ vitonhd_train_tagged.json not found")
            return False
        
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        return False

def test_streamlit_app():
    """Test Streamlit app loading"""
    print("🧪 Testing Streamlit App Loading...")
    
    try:
        # Import the main app components
        from enhanced_fashion_app import EnhancedFashionRecommender
        
        # Test app initialization
        app = EnhancedFashionRecommender()
        print(f"✅ App initialized with {len(app.metadata)} metadata items")
        print(f"✅ Template images: {len(app.template_images)}")
        print(f"✅ GAT model loaded: {app.gat_recommender.model_loaded}")
        
        # Test text recommendation
        recommendations = app.recommend_by_text("blue shirt", top_k=3)
        print(f"✅ Text recommendation successful! Found {len(recommendations)} recommendations")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Comprehensive Fashion Recommender Tests\n")
    
    tests = [
        ("GAT Architecture", test_gat_architecture),
        ("Enhanced Integration", test_enhanced_integration),
        ("Dataset Loading", test_dataset_loading),
        ("Streamlit App", test_streamlit_app),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Test")
        print('='*50)
        
        result = test_func()
        results.append((test_name, result))
        
        if result:
            print(f"✅ {test_name} Test PASSED")
        else:
            print(f"❌ {test_name} Test FAILED")
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The system is ready to use.")
        print("\nTo start the application:")
        print("streamlit run enhanced_fashion_app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
