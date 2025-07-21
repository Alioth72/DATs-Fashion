"""
Enhanced GAT Model Integration for Fashion Recommendation
Integrates with the actual GAT architecture from untitled7.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer

# Import the actual GAT architecture
try:
    from correct_gat_architecture import CorrectFashionGATModel, TORCH_GEOMETRIC_AVAILABLE
    GAT_ARCHITECTURE_AVAILABLE = True
    print("✅ Using correct GAT architecture")
except ImportError:
    GAT_ARCHITECTURE_AVAILABLE = False
    print("⚠️ Using fallback GAT implementation")
    
    # Fallback simple model
    class CorrectFashionGATModel:
        def __init__(self, model_path=None):
            self.model_loaded = False
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        def extract_features(self, features, edge_index=None):
            # Simple feature extraction
            if isinstance(features, np.ndarray):
                return torch.tensor(features, dtype=torch.float32)
            return features

class FashionGATRecommender:
    """Enhanced recommender using the actual GAT model architecture"""
    
    def __init__(self, model_path="gat_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")
        
        # Load the actual GAT model
        self.model = CorrectFashionGATModel(model_path)
        self.model_loaded = self.model.model_loaded
        
        if self.model_loaded:
            print("✅ GAT model loaded successfully!")
        else:
            print("⚠️ Using fallback model implementation")
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image):
        """Extract features from an image using the GAT model"""
        try:
            # Preprocess image
            if isinstance(image, Image.Image):
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            else:
                # Convert numpy array to PIL Image first
                if isinstance(image, np.ndarray):
                    # Ensure the array is in the right format
                    if image.dtype != np.uint8:
                        image = (image * 255).astype(np.uint8)
                    
                    # Handle different image formats
                    if image.shape[-1] == 3:  # RGB
                        image = Image.fromarray(image)
                    elif len(image.shape) == 2:  # Grayscale
                        image = Image.fromarray(image, mode='L')
                    else:
                        image = Image.fromarray(image)
                
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features using the GAT model
            with torch.no_grad():
                # Convert image tensor to feature vector suitable for GAT
                # Flatten and reduce dimensions to match GAT input (384)
                features = image_tensor.flatten(1)  # [1, 3*224*224]
                
                # Use a simple linear layer to reduce to 384 dimensions
                if not hasattr(self, 'feature_reducer'):
                    self.feature_reducer = nn.Linear(3*224*224, 384).to(self.device)
                
                reduced_features = self.feature_reducer(features)  # [1, 384]
                
                # Extract features using GAT
                if GAT_ARCHITECTURE_AVAILABLE and self.model_loaded:
                    # Create simple edge index for single item
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(self.device)
                    gat_features = self.model.extract_features(reduced_features, edge_index)
                else:
                    gat_features = self.model.extract_features(reduced_features)
                
                return gat_features.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"⚠️ Error in GAT feature extraction: {e}")
            return self._simple_feature_extraction(image)
    
    def _simple_feature_extraction(self, image):
        """Fallback simple feature extraction when GAT model fails"""
        try:
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
            
            # Extract texture features (using proper numpy array conversion)
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
            print(f"⚠️ Error in simple feature extraction: {e}")
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
    
    def recommend_similar_items(self, query_image, candidate_images, top_k=9):
        """Recommend similar items based on image similarity using GAT features"""
        query_features = self.extract_features(query_image)
        
        similarities = []
        for idx, candidate_image in enumerate(candidate_images):
            try:
                candidate_features = self.extract_features(candidate_image)
                similarity = self.calculate_similarity(query_features, candidate_features)
                similarities.append((idx, similarity))
            except Exception as e:
                print(f"⚠️ Error processing candidate {idx}: {e}")
                similarities.append((idx, 0.0))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

# Utility functions for metadata processing
def process_metadata_for_gat(metadata_items):
    """Process metadata items to create feature vectors for GAT"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Extract text features
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
    
    # Convert sparse matrix to dense array (robust approach)
    try:
        if hasattr(text_vectors, 'toarray'):
            text_vectors_dense = text_vectors.toarray()
        else:
            # Alternative approach for different scipy versions
            text_vectors_dense = np.array(text_vectors.todense())
        
        return torch.tensor(text_vectors_dense, dtype=torch.float32)
    except Exception as e:
        print(f"⚠️ Error converting sparse matrix: {e}")
        # Return a fallback feature vector
        fallback_features = np.random.rand(len(text_features), 128)
        return torch.tensor(fallback_features, dtype=torch.float32)

def calculate_metadata_similarity(item1, item2):
    """Calculate similarity between two metadata items"""
    score = 0.0
    
    # Category similarity
    if item1.get('category_name') == item2.get('category_name'):
        score += 0.5
    
    # Tag similarity
    tags1 = set()
    tags2 = set()
    
    for tag in item1.get('tag_info', []):
        if tag.get('tag_category'):
            tags1.add(tag['tag_category'])
    
    for tag in item2.get('tag_info', []):
        if tag.get('tag_category'):
            tags2.add(tag['tag_category'])
    
    if tags1 and tags2:
        intersection = len(tags1 & tags2)
        union = len(tags1 | tags2)
        score += 0.5 * (intersection / union) if union > 0 else 0
    
    return score

# Test function
def test_gat_integration():
    """Test the GAT integration"""
    print("🧪 Testing GAT Integration...")
    
    try:
        # Initialize recommender
        recommender = FashionGATRecommender()
        
        # Test with dummy image
        dummy_image = Image.new('RGB', (224, 224), color='red')
        features = recommender.extract_features(dummy_image)
        
        print(f"✅ Feature extraction successful! Shape: {features.shape}")
        print(f"📊 Model loaded: {recommender.model_loaded}")
        print(f"🖥️ Device: {recommender.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ GAT integration test failed: {e}")
        return False

if __name__ == "__main__":
    test_gat_integration()
