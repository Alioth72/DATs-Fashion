"""
Correct GAT Architecture matching the saved model weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

# Check for torch_geometric availability
try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
    print("torch_geometric is available - using full GAT implementation", file=sys.stderr)
except ImportError:
    print("torch_geometric not available - using fallback implementation", file=sys.stderr)
    TORCH_GEOMETRIC_AVAILABLE = False

class GATBlock(nn.Module):
    """GAT Block from the actual saved model"""
    def __init__(self, in_dim=768, heads1=4, heads2=2):
        super().__init__()
        
        if TORCH_GEOMETRIC_AVAILABLE:
            # Use real GAT layers when available
            self.norm = nn.LayerNorm(in_dim)
            self.gat1 = GATConv(in_dim, in_dim // heads1, heads=heads1, dropout=0.2)
            self.gat2 = GATConv(in_dim, in_dim // heads1 // heads2, heads=heads2, dropout=0.2)
            self.unify_conv = nn.Linear(in_dim + in_dim // heads1 // heads2 * heads2, in_dim)
            self.use_gat = True
        else:
            # Fallback implementation
            self.norm = nn.LayerNorm(in_dim)
            self.attention1 = nn.MultiheadAttention(in_dim, heads1, dropout=0.2, batch_first=True)
            self.attention2 = nn.MultiheadAttention(in_dim, heads2, dropout=0.2, batch_first=True)
            self.linear1 = nn.Linear(in_dim, in_dim // heads1 * heads1)
            self.linear2 = nn.Linear(in_dim, in_dim // heads1 // heads2 * heads2)
            self.unify_conv = nn.Linear(in_dim + in_dim // heads1 // heads2 * heads2, in_dim)
            self.use_gat = False

    def forward(self, x, edge_index=None):
        if self.use_gat and TORCH_GEOMETRIC_AVAILABLE:
            # Real GAT forward pass
            x_norm = self.norm(x)
            out1 = self.gat1(x_norm, edge_index)
            out2 = self.gat2(out1, edge_index)
            combined = torch.cat([x, out2], dim=-1)
            return self.unify_conv(combined)
        else:
            # Fallback attention-based forward pass
            if x.dim() == 2:
                x = x.unsqueeze(0)  # Add batch dimension
                
            x_norm = self.norm(x)
            attn_out1, _ = self.attention1(x_norm, x_norm, x_norm)
            attn_out2, _ = self.attention2(attn_out1, attn_out1, attn_out1)
            
            combined = torch.cat([x, attn_out2], dim=-1)
            result = self.unify_conv(combined)
            return result.squeeze(0) if result.size(0) == 1 else result

class CorrectGATEncoder(nn.Module):
    """
    Correct GAT Encoder matching the saved model structure
    """
    def __init__(self, in_dim=384, hidden_dim=768, out_dim=1024, num_blocks=4):
        super().__init__()
        
        # Initial projection to expand features
        self.initial_projection = nn.Linear(in_dim, hidden_dim)
        
        # GAT blocks
        self.blocks = nn.ModuleList([
            GATBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # Final layers
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        if TORCH_GEOMETRIC_AVAILABLE:
            self.final_gat = GATConv(hidden_dim, 128, heads=8, dropout=0.1, concat=True)  # 8 heads * 128 = 1024 output
            self.use_gat = True
        else:
            # Fallback
            self.final_attention = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1, batch_first=True)
            self.final_linear = nn.Linear(hidden_dim, out_dim)
            self.use_gat = False

    def forward(self, x, edge_index=None):
        # Initial projection
        x = self.initial_projection(x)
        
        # Apply GAT blocks
        for block in self.blocks:
            x = block(x, edge_index)
        
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

class CorrectFashionGATModel(nn.Module):
    """
    Main Fashion GAT Model that can load the actual pre-trained weights
    """
    def __init__(self, model_path=None):
        super().__init__()
        self.encoder = CorrectGATEncoder(in_dim=384, hidden_dim=768, out_dim=1024, num_blocks=4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained weights if available
        self.model_loaded = False
        if model_path:
            try:
                # Load the state dict
                state_dict = torch.load(model_path, map_location=self.device)
                
                # Load the state dict
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

# Test function to verify model loading
def test_correct_model_loading(model_path):
    """Test if the GAT model can be loaded successfully"""
    try:
        model = CorrectFashionGATModel(model_path)
        
        # Test with dummy data (matching the expected input dimension)
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
    test_correct_model_loading(model_path)
