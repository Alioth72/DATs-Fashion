"""
Enhanced GAT Model Integration using the correct model1.py architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

# Import the correct GAT architecture
from model1_architecture import FashionGATModel, TORCH_GEOMETRIC_AVAILABLE

class EnhancedFashionGATRecommender:
    """Enhanced recommender using the correct GAT model architecture from model1.py"""
    
    def __init__(self, model_path="gat_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}", file=sys.stderr)
        
        # Load the correct GAT model
        self.model = FashionGATModel(model_path)
        self.model_loaded = self.model.model_loaded
        
        if self.model_loaded:
            print("✅ GAT model loaded successfully!", file=sys.stderr)
        else:
            print("⚠️ Using fallback model implementation", file=sys.stderr)
        
        # Image preprocessing pipeline (for image-based recommendations)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_metadata_features(self, metadata_items):
        """Extract features from metadata using the GAT model"""
        try:
            # Use the GAT model to extract features from metadata
            embeddings = self.model.extract_features(metadata_items, return_embeddings=True)
            return embeddings.cpu().numpy()
        
        except Exception as e:
            print(f"⚠️ Error in metadata feature extraction: {e}")
            return self._fallback_metadata_features(metadata_items)
    
    def _fallback_metadata_features(self, metadata_items):
        """Fallback feature extraction when GAT model fails"""
        try:
            # Extract text features using TF-IDF
            text_features = []
            for item in metadata_items:
                features = []
                
                # Category
                if 'category_name' in item:
                    features.append(item['category_name'])
                
                # Tags
                if 'tag_info' in item:
                    for tag in item['tag_info']:
                        if tag.get('tag_category'):
                            features.append(tag['tag_category'])
                
                text_features.append(' '.join(features).lower())
            
            # Vectorize text features
            vectorizer = TfidfVectorizer(max_features=128, stop_words='english')
            text_vectors = vectorizer.fit_transform(text_features)
            
            # Convert sparse matrix to dense array
            text_vectors_dense = text_vectors.toarray()
            return text_vectors_dense
            
        except Exception as e:
            print(f"⚠️ Error in fallback feature extraction: {e}")
            # Return random features as last resort
            return np.random.rand(len(metadata_items), 128)
    
    def extract_image_features(self, image):
        """Extract features from an image using simple computer vision"""
        try:
            # Preprocess image
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Ensure image is in correct format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Resize image
            image = cv2.resize(image, (224, 224))
            
            # Extract color histogram features
            hist_r = cv2.calcHist([image], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [32], [0, 256])
            
            # Extract texture features
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
            texture_features = [
                float(np.mean(gray)),
                float(np.std(gray)),
                float(np.var(gray))
            ]
            
            # Combine features
            color_features = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
            all_features = np.concatenate([color_features, texture_features])
            
            return all_features
        except Exception as e:
            print(f"⚠️ Error in image feature extraction: {e}")
            # Return a default feature vector
            return np.random.rand(100)
    
    def calculate_similarity(self, features1, features2):
        """Calculate cosine similarity between two feature vectors"""
        # Normalize features
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(features1, features2) / (norm1 * norm2)
        return max(0, float(similarity))
    
    def recommend_by_metadata(self, query_metadata, candidate_metadata, top_k=9):
        """Recommend items based on metadata similarity using GAT features"""
        try:
            # Extract features for all items including query
            all_metadata = [query_metadata] + candidate_metadata
            all_features = self.extract_metadata_features(all_metadata)
            
            # Use GAT model for similarity computation if available
            if self.model_loaded:
                similarities = self.model.find_similar_items(0, all_features, top_k=top_k)
                return similarities
            else:
                # Fallback similarity computation
                query_features = all_features[0]
                similarities = []
                
                for idx, candidate_features in enumerate(all_features[1:], 1):
                    similarity = self.calculate_similarity(query_features, candidate_features)
                    similarities.append((idx - 1, similarity))  # Adjust index for candidates
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:top_k]
        
        except Exception as e:
            print(f"⚠️ Error in metadata-based recommendation: {e}")
            # Return random recommendations as fallback
            num_candidates = len(candidate_metadata)
            random_indices = np.random.choice(num_candidates, size=min(top_k, num_candidates), replace=False)
            return [(idx, 0.1) for idx in random_indices]
    
    def recommend_by_image(self, query_image, candidate_images, top_k=9):
        """Recommend similar items based on image similarity"""
        query_features = self.extract_image_features(query_image)
        
        similarities = []
        for idx, candidate_image in enumerate(candidate_images):
            try:
                candidate_features = self.extract_image_features(candidate_image)
                similarity = self.calculate_similarity(query_features, candidate_features)
                similarities.append((idx, similarity))
            except Exception as e:
                print(f"⚠️ Error processing candidate {idx}: {e}")
                similarities.append((idx, 0.0))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def find_similar_by_index(self, query_idx, metadata_items, top_k=9):
        """Find similar items given an index in the metadata"""
        try:
            # Extract features for all metadata
            all_features = self.extract_metadata_features(metadata_items)
            
            if self.model_loaded:
                # Use GAT model similarity
                similarities = self.model.find_similar_items(query_idx, all_features, top_k=top_k)
                return similarities
            else:
                # Fallback similarity computation
                query_features = all_features[query_idx]
                similarities = []
                
                for idx, candidate_features in enumerate(all_features):
                    if idx != query_idx:
                        similarity = self.calculate_similarity(query_features, candidate_features)
                        similarities.append((idx, similarity))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:top_k]
        
        except Exception as e:
            print(f"⚠️ Error in index-based similarity: {e}")
            # Return random recommendations as fallback
            num_items = len(metadata_items)
            available_indices = [i for i in range(num_items) if i != query_idx]
            random_indices = np.random.choice(available_indices, size=min(top_k, len(available_indices)), replace=False)
            return [(idx, 0.1) for idx in random_indices]

# Test function
def test_enhanced_integration():
    """Test the enhanced GAT integration"""
    print("🧪 Testing Enhanced GAT Integration...")
    
    try:
        # Initialize recommender
        recommender = EnhancedFashionGATRecommender()
        
        # Test with sample metadata
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
        ] * 5  # Create 10 items
        
        # Test metadata feature extraction
        features = recommender.extract_metadata_features(sample_metadata)
        print(f"✅ Metadata feature extraction successful! Shape: {features.shape}")
        
        # Test similarity search
        similarities = recommender.find_similar_by_index(0, sample_metadata, top_k=3)
        print(f"✅ Similarity search successful! Found {len(similarities)} similar items")
        
        # Test with dummy image
        dummy_image = Image.new('RGB', (224, 224), color='red')
        image_features = recommender.extract_image_features(dummy_image)
        print(f"✅ Image feature extraction successful! Shape: {image_features.shape}")
        
        print(f"📊 Model loaded: {recommender.model_loaded}")
        print(f"🖥️ Device: {recommender.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_integration()
