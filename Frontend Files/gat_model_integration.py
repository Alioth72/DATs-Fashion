import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# Try to import torch_geometric, but handle if it's not available
try:
    from torch_geometric.nn import GATConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("PyTorch Geometric not available. Using fallback implementation.")
    TORCH_GEOMETRIC_AVAILABLE = False

class SimplifiedGATModel(nn.Module):
    """Simplified GAT-like model without PyTorch Geometric dependency"""
    
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128, num_heads=8):
        super(SimplifiedGATModel, self).__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Simplified attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=num_heads, 
            dropout=0.1,
            batch_first=True
        )
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(256 * 7 * 7, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Classification/Similarity layers
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )
        
    def forward(self, x):
        # Apply feature extraction if input is image
        if len(x.shape) == 4:  # Image input
            x = self.feature_extractor(x)
            x = x.view(x.size(0), -1)
            x = self.feature_processor(x)
        
        # Apply attention mechanism (simplified)
        x = x.unsqueeze(0) if len(x.shape) == 1 else x
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(0) if attn_output.shape[0] == 1 else attn_output
        
        # Final classification
        if len(x.shape) == 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        elif len(x.shape) == 3:
            x = x.mean(dim=1)  # Global average pooling
        
        x = self.classifier(x)
        return x
    
    def extract_image_features(self, image):
        """Extract features from a single image"""
        with torch.no_grad():
            # Preprocess image
            if isinstance(image, Image.Image):
                image = transforms.ToTensor()(image).unsqueeze(0)
            elif isinstance(image, np.ndarray):
                image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # Extract features using CNN backbone
            features = self.feature_extractor(image)
            features = features.view(features.size(0), -1)
            features = self.feature_processor(features)
            return features

# Keep the original class name for backward compatibility
if TORCH_GEOMETRIC_AVAILABLE:
    class GATFashionModel(nn.Module):
        """Graph Attention Network for Fashion Recommendation"""
        
        def __init__(self, input_dim=512, hidden_dim=256, output_dim=128, num_heads=8):
            super(GATFashionModel, self).__init__()
            
            # Feature extraction layers
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7))
            )
            
            # GAT layers
            self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.1)
            self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=0.1)
            
            # Classification/Similarity layers
            self.classifier = nn.Sequential(
                nn.Linear(output_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32)
            )
            
        def forward(self, x, edge_index, batch=None):
            # Apply GAT layers
            x = F.dropout(x, p=0.1, training=self.training)
            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.1, training=self.training)
            x = self.gat2(x, edge_index)
            
            # Global pooling if batch is provided
            if batch is not None:
                x = global_mean_pool(x, batch)
            
            # Final classification
            x = self.classifier(x)
            return x
        
        def extract_image_features(self, image):
            """Extract features from a single image"""
            with torch.no_grad():
                # Preprocess image
                if isinstance(image, Image.Image):
                    image = transforms.ToTensor()(image).unsqueeze(0)
                elif isinstance(image, np.ndarray):
                    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                
                # Extract features using CNN backbone
                features = self.feature_extractor(image)
                features = features.view(features.size(0), -1)
                return features
else:
    # Use simplified model when PyTorch Geometric is not available
    GATFashionModel = SimplifiedGATModel

class FashionGATRecommender:
    """Enhanced recommender using GAT model"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GATFashionModel()
        
        # Load pre-trained weights if available
        if model_path and self.device.type == 'cuda':
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model_loaded = True
                print("✅ GAT model loaded successfully!")
            except Exception as e:
                print(f"⚠️ Could not load GAT model: {e}")
                self.model_loaded = False
        else:
            print("ℹ️ Using simplified model (no GPU or model file not found)")
            self.model_loaded = False
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image):
        """Extract features from an image using the GAT model"""
        if not self.model_loaded:
            # Fallback to simple feature extraction
            return self._simple_feature_extraction(image)
        
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
                    image = Image.fromarray(image)
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                if TORCH_GEOMETRIC_AVAILABLE and hasattr(self.model, 'extract_image_features'):
                    features = self.model.extract_image_features(image_tensor)
                else:
                    # Use simplified forward pass
                    features = self.model(image_tensor)
                return features.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"⚠️ Error in GAT feature extraction: {e}")
            return self._simple_feature_extraction(image)
    
    def _simple_feature_extraction(self, image):
        """Fallback simple feature extraction when GAT model is not available"""
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
            
            # Extract texture features
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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
        """Recommend similar items based on image similarity"""
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

def create_graph_data(image_features, metadata_features=None):
    """Create graph data structure for GAT model"""
    if not TORCH_GEOMETRIC_AVAILABLE:
        # Return simple tensor when torch_geometric is not available
        if metadata_features is not None:
            return torch.cat([image_features, metadata_features], dim=1)
        return image_features
    
    try:
        from torch_geometric.data import Data
        
        # Create node features (image features + metadata if available)
        if metadata_features is not None:
            node_features = torch.cat([image_features, metadata_features], dim=1)
        else:
            node_features = image_features
        
        # Create a simple graph structure (fully connected for similarity)
        num_nodes = node_features.size(0)
        edge_index = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        
        return Data(x=node_features, edge_index=edge_index)
    
    except ImportError:
        # Fallback when torch_geometric is not available
        if metadata_features is not None:
            return torch.cat([image_features, metadata_features], dim=1)
        return image_features

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
    
    # Convert sparse matrix to dense array
    try:
        # Try different methods for sparse matrix conversion
        if hasattr(text_vectors, 'toarray'):
            text_vectors_dense = text_vectors.toarray()
        elif hasattr(text_vectors, 'todense'):
            import numpy as np
            text_vectors_dense = np.array(text_vectors.todense())
        else:
            # Fallback: convert to dense using numpy
            import numpy as np
            text_vectors_dense = np.array(text_vectors)
        
        return torch.tensor(text_vectors_dense, dtype=torch.float32)
    except Exception as e:
        print(f"⚠️ Error converting sparse matrix: {e}")
        # Return a fallback feature vector
        import numpy as np
        fallback_features = np.random.rand(len(text_features), 128)
        return torch.tensor(fallback_features, dtype=torch.float32)
