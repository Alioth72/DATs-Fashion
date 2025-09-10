import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# CONFIGS
# Using user-provided paths.
json_path = "vitonhd_test_tagged.json" 
cloth_dir = "/content/unzipped_drive/clothes_tryon_dataset/test/cloth"
MODEL_SAVE_PATH = "gat_model.pth" # Path to save/load the trained model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Create a dummy JSON file for demonstration if it doesn't exist ---
try:
    with open(json_path, 'r') as f:
        pass
except FileNotFoundError:
    print(f"'{json_path}' not found. Creating a dummy file for demonstration.")
    dummy_data = {
        "data": [
            {"file_name": f"image_{i}.jpg", "tag_info": [
                {"tag_category": "CategoryA", "tag_name": f"Name{i % 3}"},
                {"tag_category": "CategoryB", "tag_name": f"Detail{i % 5}"}
            ]} for i in range(100)
        ]
    }
    with open(json_path, 'w') as f:
        json.dump(dummy_data, f)


embedding_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# STEP 1: Load tags from JSON file
with open(json_path, 'r') as f:
    full_json = json.load(f)

data_vectors = []
filenames = []

for item in full_json["data"]:
    tags = item.get("tag_info", [])
    if not tags:
        continue
    tag_text = ", ".join([f"{tag['tag_category']}: {tag['tag_name']}" for tag in tags])
    data_vectors.append(tag_text)
    filenames.append(item["file_name"])

print(f"✅ Loaded {len(data_vectors)} tag sets from JSON")

# STEP 2: Encode with MiniLM
with torch.no_grad():
    encoded_vectors = embedding_model.encode(data_vectors, convert_to_tensor=True, show_progress_bar=True).to(device)
print(f"✅ Embedding shape: {encoded_vectors.shape}")

# STEP 3: Build similarity graph
def build_graph(encoded_vectors, k=5):
    """Builds a k-NN graph from node embeddings."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(encoded_vectors.cpu().numpy())
    distances, indices = nbrs.kneighbors(encoded_vectors.cpu().numpy())
    edge_index = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:
            edge_index.append([i, neighbor])
            edge_index.append([neighbor, i])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)

edge_index = build_graph(encoded_vectors, k=5)
print(f"✅ Graph edges shape: {edge_index.shape}")

# STEP 4: Extremely Complex GAT model
class ParallelGATBlock(nn.Module):
    def __init__(self, channels, head_config_1, head_config_2):
        super().__init__()
        heads1, out_channels1 = head_config_1
        heads2, out_channels2 = head_config_2
        self.norm = nn.LayerNorm(channels)
        self.gat1 = GATConv(channels, out_channels1, heads=heads1, concat=True)
        self.gat2 = GATConv(channels, out_channels2, heads=heads2, concat=True)
        concat_dim = (out_channels1 * heads1) + (out_channels2 * heads2)
        self.unify_conv = nn.Linear(concat_dim, channels)
        
    def forward(self, x, edge_index):
        residual = x
        x_norm = self.norm(x)
        out1 = self.gat1(x_norm, edge_index)
        out2 = self.gat2(x_norm, edge_index)
        x_cat = torch.cat([out1, out2], dim=-1)
        x_cat = F.elu(x_cat)
        unified_output = self.unify_conv(x_cat)
        return residual + unified_output

class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=3, projection_factor=2):
        super().__init__()
        projected_channels = in_channels * projection_factor
        self.initial_projection = nn.Linear(in_channels, projected_channels)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = ParallelGATBlock(
                channels=projected_channels,
                head_config_1=(4, projected_channels // 4),
                head_config_2=(2, projected_channels // 8)
            )
            self.blocks.append(block)
        self.final_norm = nn.LayerNorm(projected_channels)
        self.final_gat = GATConv(projected_channels, out_channels, heads=8, concat=False, dropout=0.3)

    def forward(self, x, edge_index):
        x = self.initial_projection(x)
        x = F.elu(x)
        for block in self.blocks:
            x = block(x, edge_index)
            x = F.elu(x)
        x = self.final_norm(x)
        x = self.final_gat(x, edge_index)
        return x

# STEP 5: Triplet mining
def get_triplets(embeddings, margin=0.2):
    """A simple but effective online triplet mining strategy."""
    triplets = []
    num_samples = embeddings.shape[0]
    pdist = nn.PairwiseDistance()

    for anchor_idx in range(num_samples):
        anchor_emb = embeddings[anchor_idx].unsqueeze(0)
        dists = pdist(anchor_emb, embeddings)
        dists[anchor_idx] = float('inf')
        positive_idx = torch.argmin(dists).item()
        negative_indices = torch.where(dists > margin)[0]
        
        if len(negative_indices) > 0:
            rand_neg_idx = torch.randint(len(negative_indices), (1,)).item()
            negative_idx = negative_indices[rand_neg_idx].item()
            triplets.append((anchor_idx, positive_idx, negative_idx))
            
    return triplets


# STEP 6: Training or Loading the Model
gat = GATEncoder(in_channels=384, out_channels=128, n_blocks=4).to(device)
data = Data(x=encoded_vectors, edge_index=edge_index)

# Check if a pre-trained model exists
if os.path.exists(MODEL_SAVE_PATH):
    print(f"✅ Found pre-trained model at '{MODEL_SAVE_PATH}'. Loading weights...")
    gat.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print("✅ Model loaded successfully. Skipping training.")
else:
    print(f"⚠️ No pre-trained model found. Starting training...")
    optimizer = torch.optim.Adam(gat.parameters(), lr=5e-5, weight_decay=1e-5)
    triplet_loss_fn = nn.TripletMarginLoss(margin=0.5)

    for epoch in range(100):
        gat.train()
        optimizer.zero_grad()
        
        z = gat(data.x, data.edge_index)
        triplets = get_triplets(z)
        
        if not triplets:
            print(f"⚠️ No valid triplets found in epoch {epoch+1}, skipping...")
            continue

        a, p, n = zip(*triplets)
        loss = triplet_loss_fn(z[list(a)], z[list(p)], z[list(n)])
        loss.backward()
        optimizer.step()
        
        print(f"✅ Epoch {epoch + 1}/100 - Loss: {loss.item():.4f} - Triplets found: {len(triplets)}")

    # Save the trained model
    torch.save(gat.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Training complete. Model saved to '{MODEL_SAVE_PATH}'.")


# STEP 7: Final embeddings
print("\n✅ Generating final embeddings...")
gat.eval()
with torch.no_grad():
    final_embeddings = gat(data.x, data.edge_index).cpu()

print(f"✅ Final embeddings generated with shape: {final_embeddings.shape}")


# STEP 8: Visualize Similarity Search Results
print("\n🖼️ Visualizing similarity search results...")

assert len(filenames) == final_embeddings.shape[0], "Mismatch in filenames and embeddings"

emb_norm = F.normalize(final_embeddings, dim=1)
sim_matrix = emb_norm @ emb_norm.T

num_queries = 10
top_k = 5
query_indices = np.random.choice(len(filenames), size=num_queries, replace=False)

fig, axes = plt.subplots(num_queries, top_k + 1, figsize=(15, 3 * num_queries))
if num_queries == 1:
    axes = axes.reshape(1, -1)

for row, query_idx in enumerate(query_indices):
    sims = sim_matrix[query_idx]
    top_indices = torch.topk(sims, k=top_k + 1).indices.tolist()

    for col, img_idx in enumerate(top_indices):
        if img_idx == query_idx:
            title = "Query"
        else:
            rank = top_indices.index(img_idx)
            title = f"Top {rank}"

        img_path = os.path.join(cloth_dir, filenames[img_idx])
        
        try:
            img = Image.open(img_path).convert("RGB")
            axes[row, col].imshow(img)
            axes[row, col].set_title(title, fontsize=10)
        except FileNotFoundError:
            axes[row, col].text(0.5, 0.5, 'Image\nnot found', ha='center', va='center')
            print(f"⚠️ Warning: Image not found at {img_path}")
            
        axes[row, col].axis('off')

plt.tight_layout()
plt.show()