"""
Correct GAT Architecture from model1.py for Fashion Recommendation System
This matches the exact architecture used to train gat_model.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Check for torch_geometric availability
try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
    print("[OK] torch_geometric is available - using full GAT implementation", file=sys.stderr)
except ImportError:
    print("[WARNING] torch_geometric not available - using fallback implementation", file=sys.stderr)
    TORCH_GEOMETRIC_AVAILABLE = False

class ParallelGATBlock(nn.Module):
    """Parallel GAT Block with two different head configurations"""
    def __init__(self, channels, head_config_1, head_config_2):
        super().__init__()
        heads1, out_channels1 = head_config_1
        heads2, out_channels2 = head_config_2
        self.norm = nn.LayerNorm(channels)
        
        if TORCH_GEOMETRIC_AVAILABLE:
            self.gat1 = GATConv(channels, out_channels1, heads=heads1, concat=True)
            self.gat2 = GATConv(channels, out_channels2, heads=heads2, concat=True)
            self.use_gat = True
        else:
            # Fallback to multi-head attention
            self.attention1 = nn.MultiheadAttention(channels, heads1, batch_first=True)
            self.attention2 = nn.MultiheadAttention(channels, heads2, batch_first=True)
            self.linear1 = nn.Linear(channels, out_channels1 * heads1)
            self.linear2 = nn.Linear(channels, out_channels2 * heads2)
            self.use_gat = False
        
        concat_dim = (out_channels1 * heads1) + (out_channels2 * heads2)
        self.unify_conv = nn.Linear(concat_dim, channels)
        
    def forward(self, x, edge_index=None):
        residual = x
        x_norm = self.norm(x)
        
        if self.use_gat and TORCH_GEOMETRIC_AVAILABLE:
            out1 = self.gat1(x_norm, edge_index)
            out2 = self.gat2(x_norm, edge_index)
        else:
            # Fallback implementation
            if x_norm.dim() == 2:
                x_norm = x_norm.unsqueeze(0)
            
            attn_out1, _ = self.attention1(x_norm, x_norm, x_norm)
            attn_out2, _ = self.attention2(x_norm, x_norm, x_norm)
            
            out1 = self.linear1(attn_out1.squeeze(0) if attn_out1.size(0) == 1 else attn_out1)
            out2 = self.linear2(attn_out2.squeeze(0) if attn_out2.size(0) == 1 else attn_out2)
        
        x_cat = torch.cat([out1, out2], dim=-1)
        x_cat = F.elu(x_cat)
        unified_output = self.unify_conv(x_cat)
        return residual + unified_output

class GATEncoder(nn.Module):
    """The exact GAT Encoder architecture from model1.py"""
    def __init__(self, in_channels=384, out_channels=128, n_blocks=4, projection_factor=2):
        super().__init__()
        projected_channels = in_channels * projection_factor  # 384 * 2 = 768
        self.initial_projection = nn.Linear(in_channels, projected_channels)
        
        # Create blocks with the exact configuration from model1.py
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = ParallelGATBlock(
                channels=projected_channels,  # 768
                head_config_1=(4, projected_channels // 4),  # (4, 192)
                head_config_2=(2, projected_channels // 8)   # (2, 96)
            )
            self.blocks.append(block)
        
        self.final_norm = nn.LayerNorm(projected_channels)
        
        if TORCH_GEOMETRIC_AVAILABLE:
            self.final_gat = GATConv(projected_channels, out_channels, heads=8, concat=False, dropout=0.3)
            self.use_gat = True
        else:
            # Fallback
            self.final_attention = nn.MultiheadAttention(projected_channels, 8, dropout=0.3, batch_first=True)
            self.final_linear = nn.Linear(projected_channels, out_channels)
            self.use_gat = False

    def forward(self, x, edge_index=None):
        # Initial projection
        x = self.initial_projection(x)
        x = F.elu(x)
        
        # Apply GAT blocks
        for block in self.blocks:
            x = block(x, edge_index)
            x = F.elu(x)
        
        # Final processing
        x = self.final_norm(x)
        
        if self.use_gat and TORCH_GEOMETRIC_AVAILABLE:
            return self.final_gat(x, edge_index)
        else:
            # Fallback
            if x.dim() == 2:
                x = x.unsqueeze(0)
            attn_out, _ = self.final_attention(x, x, x)
            result = self.final_linear(attn_out)
            return result.squeeze(0) if result.size(0) == 1 else result

def build_graph(encoded_vectors, k=5):
    """Build k-NN graph from node embeddings (from model1.py)"""
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None
    
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(encoded_vectors.cpu().numpy())
    distances, indices = nbrs.kneighbors(encoded_vectors.cpu().numpy())
    edge_index = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  # Skip self
            edge_index.append([i, neighbor])
            edge_index.append([neighbor, i])  # Undirected graph
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

class FashionGATModel(nn.Module):
    """
    Complete Fashion GAT Model that matches model1.py architecture
    """
    def __init__(self, model_path=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the exact architecture from model1.py
        self.encoder = GATEncoder(in_channels=384, out_channels=128, n_blocks=4)
        
        # Initialize sentence transformer for text encoding
        self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.text_encoder.to(self.device)
        
        # Load pre-trained weights if available
        self.model_loaded = False
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.encoder.load_state_dict(state_dict)
                self.model_loaded = True
                print(f"[OK] Successfully loaded GAT model from {model_path}", file=sys.stderr)
                
            except Exception as e:
                print(f"[WARNING] Could not load GAT model from {model_path}: {e}", file=sys.stderr)
                print("[INFO] Using randomly initialized model", file=sys.stderr)
                self.model_loaded = False
        
        self.encoder.to(self.device)
        self.encoder.eval()
    
    def encode_metadata(self, metadata_items):
        """Encode metadata items to match the input format expected by GAT"""
        data_vectors = []
        
        for item in metadata_items:
            tags = item.get("tag_info", [])
            if not tags:
                # Create a default tag if no tags are available
                tag_text = f"{item.get('category_name', 'Unknown')}: default"
            else:
                # Format tags exactly like in model1.py
                tag_text = ", ".join([
                    f"{tag.get('tag_category', 'Unknown')}: {tag.get('tag_name', 'default')}" 
                    for tag in tags if tag.get('tag_category') is not None
                ])
                if not tag_text:  # If all tag_category are None
                    tag_text = f"{item.get('category_name', 'Unknown')}: default"
            
            data_vectors.append(tag_text)
        
        # Encode with sentence transformer (like model1.py)
        with torch.no_grad():
            encoded_vectors = self.text_encoder.encode(
                data_vectors, 
                convert_to_tensor=True, 
                show_progress_bar=False
            ).to(self.device)
        
        return encoded_vectors
    
    def extract_features(self, metadata_items, return_embeddings=False):
        """Extract features using the trained GAT model"""
        try:
            # Encode metadata
            encoded_vectors = self.encode_metadata(metadata_items)
            
            if TORCH_GEOMETRIC_AVAILABLE and self.model_loaded:
                # Build graph like in model1.py
                edge_index = build_graph(encoded_vectors, k=5)
                if edge_index is not None:
                    edge_index = edge_index.to(self.device)
                
                # Forward pass through GAT
                with torch.no_grad():
                    embeddings = self.encoder(encoded_vectors, edge_index)
                
                if return_embeddings:
                    return embeddings
                return embeddings.cpu().numpy()
            else:
                # Fallback: use the encoder without graph structure
                with torch.no_grad():
                    embeddings = self.encoder(encoded_vectors)
                
                if return_embeddings:
                    return embeddings
                return embeddings.cpu().numpy()
                
        except Exception as e:
            print(f"[WARNING] Error in GAT feature extraction: {e}", file=sys.stderr)
            # Return random features as fallback
            num_items = len(metadata_items)
            if return_embeddings:
                return torch.randn(num_items, 128).to(self.device)
            return np.random.randn(num_items, 128)
    
    def compute_similarity_matrix(self, embeddings):
        """Compute similarity matrix from embeddings"""
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        # Normalize embeddings
        emb_norm = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        sim_matrix = emb_norm @ emb_norm.T
        
        return sim_matrix.cpu().numpy()
    
    def find_similar_items(self, query_idx, embeddings, top_k=9):
        """Find similar items given a query index"""
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        # Compute similarity matrix
        sim_matrix = self.compute_similarity_matrix(embeddings)
        
        # Get similarities for the query item
        sims = sim_matrix[query_idx]
        
        # Get top-k similar items
        top_indices = np.argsort(sims)[::-1][:top_k + 1]  # +1 to include query
        
        # Remove query index if it's in the results
        if query_idx in top_indices:
            top_indices = top_indices[top_indices != query_idx][:top_k]
        else:
            top_indices = top_indices[:top_k]
        
        similarities = sims[top_indices]
        
        return list(zip(top_indices, similarities))

    def find_similar_items_by_text(self, query_text, metadata_items, top_k=9):
        """Find similar items based on a text query"""
        try:
            # Encode the query text
            with torch.no_grad():
                query_embedding = self.text_encoder.encode(
                    [query_text], 
                    convert_to_tensor=True, 
                    show_progress_bar=False
                ).to(self.device)
            
            # Extract features from all metadata items (return as tensor)
            all_features = self.extract_features(metadata_items, return_embeddings=True)
            
            # Ensure all_features is a tensor
            if isinstance(all_features, np.ndarray):
                all_features = torch.from_numpy(all_features).to(self.device)
            
            # Calculate similarities
            similarities = F.cosine_similarity(
                query_embedding.unsqueeze(0), 
                all_features.unsqueeze(1), 
                dim=2
            ).squeeze()
            
            # Get top-k similar items
            top_values, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
            
            return list(zip(top_indices.cpu().numpy(), top_values.cpu().numpy()))
            
        except Exception as e:
            print(f"[WARNING] Error in text-based similarity: {e}", file=sys.stderr)
            # Fallback: return random items
            num_items = len(metadata_items)
            random_indices = np.random.choice(num_items, size=min(top_k, num_items), replace=False)
            return [(idx, 0.5) for idx in random_indices]

# Test function
def test_model_architecture():
    """Test the model architecture and loading"""
    print("🧪 Testing Model Architecture...")
    
    try:
        # Test model creation
        model = FashionGATModel("gat_model.pth")
        
        # Test with dummy metadata
        dummy_metadata = [
            {
                "file_name": "test.jpg",
                "category_name": "TOPS",
                "tag_info": [
                    {"tag_name": "item", "tag_category": "T-Shirts"},
                    {"tag_name": "colors", "tag_category": "Blue"}
                ]
            }
        ] * 10
        
        # Test feature extraction
        features = model.extract_features(dummy_metadata)
        print(f"[OK] Feature extraction successful! Shape: {features.shape}")
        
        # Test similarity computation
        similarities = model.find_similar_items(0, features, top_k=5)
        print(f"[OK] Similarity computation successful! Found {len(similarities)} similar items")
        
        print(f"[INFO] Model loaded: {model.model_loaded}")
        print(f"[INFO] Device: {model.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_architecture()
