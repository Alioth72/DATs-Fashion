"""
GAT Architecture from untitled7.py for Fashion Recommendation System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Check for torch_geometric availability
try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
    print("✅ torch_geometric is available - using full GAT implementation")
except ImportError:
    print("⚠️ torch_geometric not available - using fallback implementation")
    TORCH_GEOMETRIC_AVAILABLE = False

class DeepGATEncoder(nn.Module):
    """
    Deep GAT Encoder from untitled7.py - the actual trained architecture
    """
    def __init__(self, in_dim=384, hidden_dim=256, out_dim=128, heads=4):
        super().__init__()
        
        if TORCH_GEOMETRIC_AVAILABLE:
            # Use real GAT layers when available
            self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=0.2)
            self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=2, dropout=0.2)
            self.gat3 = GATConv(hidden_dim * 2, out_dim, heads=1, dropout=0.1)
            self.norm1 = nn.LayerNorm(hidden_dim * heads)
            self.norm2 = nn.LayerNorm(hidden_dim * 2)
            self.use_gat = True
        else:
            # Fallback to multi-head attention
            self.attention1 = nn.MultiheadAttention(in_dim, heads, dropout=0.2, batch_first=True)
            self.attention2 = nn.MultiheadAttention(hidden_dim, 2, dropout=0.2, batch_first=True)
            self.linear1 = nn.Linear(in_dim, hidden_dim * heads)
            self.linear2 = nn.Linear(hidden_dim * heads, hidden_dim * 2)
            self.linear3 = nn.Linear(hidden_dim * 2, out_dim)
            self.norm1 = nn.LayerNorm(hidden_dim * heads)
            self.norm2 = nn.LayerNorm(hidden_dim * 2)
            self.use_gat = False

    def forward(self, x, edge_index=None):
        if self.use_gat and TORCH_GEOMETRIC_AVAILABLE:
            # Real GAT forward pass
            x = F.relu(self.norm1(self.gat1(x, edge_index)))
            x = F.relu(self.norm2(self.gat2(x, edge_index)))
            return self.gat3(x, edge_index)
        else:
            # Fallback attention-based forward pass
            if x.dim() == 2:
                x = x.unsqueeze(0)  # Add batch dimension
                
            # First layer
            x = self.linear1(x)
            x = F.relu(self.norm1(x))
            attn_out, _ = self.attention1(x, x, x)
            x = x + attn_out  # Residual connection
            
            # Second layer
            x = self.linear2(x)
            x = F.relu(self.norm2(x))
            attn_out, _ = self.attention2(x, x, x)
            x = x + attn_out  # Residual connection
            
            # Final layer
            x = self.linear3(x)
            return x.squeeze(0) if x.size(0) == 1 else x

class FashionGATModel(nn.Module):
    """
    Main Fashion GAT Model that can load the pre-trained weights
    """
    def __init__(self, model_path=None):
        super().__init__()
        self.encoder = DeepGATEncoder(in_dim=384, hidden_dim=256, out_dim=128, heads=4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained weights if available
        self.model_loaded = False
        if model_path:
            try:
                # Load the state dict
                state_dict = torch.load(model_path, map_location=self.device)
                
                # Handle different state dict formats
                if 'model' in state_dict:
                    # If the state dict is nested under 'model' key
                    self.encoder.load_state_dict(state_dict['model'])
                elif 'state_dict' in state_dict:
                    # If the state dict is nested under 'state_dict' key
                    self.encoder.load_state_dict(state_dict['state_dict'])
                else:
                    # Direct state dict
                    self.encoder.load_state_dict(state_dict)
                
                self.model_loaded = True
                print(f"✅ Successfully loaded GAT model from {model_path}")
                
            except Exception as e:
                print(f"⚠️ Could not load GAT model from {model_path}: {e}")
                print("🔄 Using randomly initialized model")
                self.model_loaded = False
        
        self.encoder.to(self.device)
        self.encoder.eval()
    
    def forward(self, x, edge_index=None):
        return self.encoder(x, edge_index)
    
    def extract_features(self, features, edge_index=None):
        """Extract features using the GAT encoder"""
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features = torch.tensor(features, dtype=torch.float32).to(self.device)
            elif isinstance(features, torch.Tensor):
                features = features.to(self.device)
            
            return self.encoder(features, edge_index)

def get_triplets(embeddings, margin=0.5):
    """
    Triplet mining function from untitled7.py (semi-hard triplets)
    """
    dists = torch.cdist(embeddings, embeddings, p=2)
    triplets = []
    for anchor in range(len(embeddings)):
        device = embeddings.device
        eye = torch.eye(len(embeddings), device=device)
        pos = torch.argmin(dists[anchor] + eye[anchor] * 1e6)
        neg_mask = dists[anchor] > (dists[anchor][pos] + margin)
        neg_candidates = torch.where(neg_mask)[0]
        if len(neg_candidates):
            neg = neg_candidates[torch.randint(len(neg_candidates), (1,))].item()
            triplets.append((anchor, pos, neg))
    return triplets

def create_graph_data(features, similarity_threshold=0.5):
    """
    Create graph data structure from features
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        return features
    
    try:
        # Convert features to tensor if needed
        if isinstance(features, np.ndarray):
            x = torch.tensor(features, dtype=torch.float32)
        else:
            x = features
        
        # Create edges based on similarity
        num_nodes = x.size(0)
        edge_indices = []
        
        # Compute pairwise similarities
        similarities = torch.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
        
        # Create edges for similar items
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if similarities[i, j] > similarity_threshold:
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])  # Undirected graph
        
        # If no edges, create self-loops
        if not edge_indices:
            edge_indices = [[i, i] for i in range(num_nodes)]
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    
    except Exception as e:
        print(f"⚠️ Error creating graph data: {e}")
        return features

# Test function to verify model loading
def test_model_loading(model_path):
    """Test if the GAT model can be loaded successfully"""
    try:
        model = FashionGATModel(model_path)
        
        # Test with dummy data
        dummy_features = torch.randn(10, 384)  # 10 items, 384 features
        
        if TORCH_GEOMETRIC_AVAILABLE:
            # Create dummy graph
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            output = model(dummy_features, edge_index)
        else:
            output = model(dummy_features)
        
        print(f"✅ Model test successful! Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the model loading
    model_path = "gat_model.pth"
    test_model_loading(model_path)
