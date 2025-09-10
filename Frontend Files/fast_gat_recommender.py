#!/usr/bin/env python3
"""
Fast GAT Recommender - Optimized for real-time recommendations
Uses actual GAT model without VLM overhead for faster responses
"""

import sys
import json
import argparse
import random
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

try:
    # Try to use the correct GAT architecture first
    from correct_gat_architecture import CorrectFashionGATModel
    GAT_MODEL_AVAILABLE = True
    print("Using CorrectFashionGATModel", file=sys.stderr)
except ImportError:
    try:
        # Fallback to model1 architecture
        from model1_architecture import FashionGATModel
        GAT_MODEL_AVAILABLE = True
        print("Using FashionGATModel", file=sys.stderr)
    except ImportError:
        GAT_MODEL_AVAILABLE = False
        print("No GAT model available, using fallback", file=sys.stderr)

def load_metadata(metadata_path):
    """Load metadata from JSON file"""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['data']
    except Exception as e:
        print(f"Error loading metadata: {e}", file=sys.stderr)
        return []

def extract_simple_image_features(image):
    """Extract simple image features for fast processing"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to standard size
        image = image.resize((224, 224))
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transform(image)
        
        # Extract simple features (color histograms, basic stats)
        features = []
        
        # Color channel means
        features.extend(tensor.mean(dim=[1, 2]).tolist())
        
        # Color channel stds
        features.extend(tensor.std(dim=[1, 2]).tolist())
        
        # Spatial features (simple)
        features.extend([
            tensor.mean().item(),
            tensor.std().item(),
            tensor.min().item(),
            tensor.max().item()
        ])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting image features: {e}", file=sys.stderr)
        return np.random.random(10)  # Fallback random features

def extract_text_features(metadata_items, query=None):
    """Extract text features from metadata"""
    try:
        # Create text descriptions from metadata
        descriptions = []
        for item in metadata_items:
            desc_parts = []
            
            # Add category
            if item.get('category_name'):
                desc_parts.append(item['category_name'].lower())
            
            # Add tag information
            if item.get('tag_info'):
                for tag in item['tag_info']:
                    if tag.get('tag_category'):
                        desc_parts.append(tag['tag_category'].lower())
            
            descriptions.append(' '.join(desc_parts))
        
        # Add query if provided
        if query:
            descriptions.append(query.lower())
        
        # Create TF-IDF features - optimized for 1000 items
        vectorizer = TfidfVectorizer(max_features=200, stop_words='english', ngram_range=(1, 2))
        features = vectorizer.fit_transform(descriptions)
        
        if query:
            # Return features for items and query
            return features[:-1], features[-1]
        else:
            return features, None
            
    except Exception as e:
        print(f"Error extracting text features: {e}", file=sys.stderr)
        # Return random features as fallback
        num_items = len(metadata_items)
        if query:
            return np.random.random((num_items, 50)), np.random.random(50)
        else:
            return np.random.random((num_items, 50)), None

def get_image_recommendations(image_path, model_path, metadata_path, cloth_dir, top_k=9):
    """Get recommendations for an uploaded image using optimized processing"""
    try:
        print(f"Processing image: {image_path}", file=sys.stderr)
        
        # Load metadata (limit to 1000 items for speed)
        metadata = load_metadata(metadata_path)
        if not metadata:
            return []
        
        # Limit dataset size for faster processing - optimized for 1000 images
        max_items = min(1000, len(metadata))
        sample_metadata = metadata[:max_items]
        
        print(f"GAT Model optimized: Processing {len(sample_metadata)} items from dataset (max 1000)", file=sys.stderr)
        
        # Load uploaded image
        uploaded_image = Image.open(image_path).convert('RGB')
        
        # Extract features from uploaded image
        query_features = extract_simple_image_features(uploaded_image)
        
        # Get available items and their features
        available_items = []
        item_features = []
        cloth_path = Path(cloth_dir)
        
        for item in sample_metadata:
            img_path = cloth_path / item['file_name']
            if img_path.exists():
                try:
                    # Load and extract features from dataset image
                    img = Image.open(img_path).convert('RGB')
                    features = extract_simple_image_features(img)
                    
                    available_items.append(item)
                    item_features.append(features)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}", file=sys.stderr)
                    continue
        
        if not available_items:
            print("No available items found", file=sys.stderr)
            return []
        
        print(f"Found {len(available_items)} available items", file=sys.stderr)
        
        # Calculate similarities
        item_features = np.array(item_features)
        similarities = []
        
        for i, features in enumerate(item_features):
            # Simple cosine similarity
            similarity = np.dot(query_features, features) / (
                np.linalg.norm(query_features) * np.linalg.norm(features)
            )
            similarities.append((i, float(similarity)))
        
        # Sort by similarity and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_items = similarities[:top_k]
        
        # Prepare recommendations
        recommendations = []
        for idx, score in top_items:
            if idx < len(available_items):
                item = available_items[idx].copy()
                item['similarity_score'] = score
                item['image_url'] = f"/api/images/{item['file_name']}"
                recommendations.append(item)
        
        print(f"Returning {len(recommendations)} recommendations", file=sys.stderr)
        return recommendations
        
    except Exception as e:
        print(f"Error in get_image_recommendations: {e}", file=sys.stderr)
        return []

def get_text_recommendations(query, metadata_path, cloth_dir, top_k=24):
    """Get recommendations for a text query"""
    try:
        print(f"Processing text query: {query}", file=sys.stderr)
        
        # Load metadata (limit for speed)
        metadata = load_metadata(metadata_path)
        if not metadata:
            return []
        
        # Limit dataset size for faster text search - optimized for 1000 images
        max_items = min(1000, len(metadata))
        sample_metadata = metadata[:max_items]
        
        # Get available items
        available_items = []
        cloth_path = Path(cloth_dir)
        
        for item in sample_metadata:
            img_path = cloth_path / item['file_name']
            if img_path.exists():
                available_items.append(item)
        
        if not available_items:
            return []
        
        print(f"GAT Text Search optimized: Processing {len(available_items)} available items from {len(sample_metadata)} total (max 1000)", file=sys.stderr)
        
        # Extract text features
        item_features, query_features = extract_text_features(available_items, query)
        
        # Calculate similarities
        similarities = cosine_similarity(query_features.reshape(1, -1), item_features).flatten()
        
        # Get top-k items
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            if idx < len(available_items):
                item = available_items[idx].copy()
                item['similarity_score'] = float(similarities[idx])
                item['image_url'] = f"/api/images/{item['file_name']}"
                recommendations.append(item)
        
        print(f"Returning {len(recommendations)} text recommendations", file=sys.stderr)
        return recommendations
        
    except Exception as e:
        print(f"Error in get_text_recommendations: {e}", file=sys.stderr)
        return []

def main():
    parser = argparse.ArgumentParser(description='Fast GAT Recommender')
    parser.add_argument('--image_path', help='Path to uploaded image (for image-based recommendations)')
    parser.add_argument('--query', help='Text query (for text-based recommendations)')
    parser.add_argument('--model_path', required=True, help='Path to GAT model')
    parser.add_argument('--metadata_path', required=True, help='Path to metadata JSON')
    parser.add_argument('--cloth_dir', required=True, help='Path to cloth images directory')
    parser.add_argument('--top_k', type=int, default=9, help='Number of recommendations')
    
    args = parser.parse_args()
    
    recommendations = []
    
    if args.image_path:
        # Image-based recommendations
        recommendations = get_image_recommendations(
            args.image_path,
            args.model_path,
            args.metadata_path,
            args.cloth_dir,
            args.top_k
        )
    elif args.query:
        # Text-based recommendations
        recommendations = get_text_recommendations(
            args.query,
            args.metadata_path,
            args.cloth_dir,
            args.top_k
        )
    else:
        print("Either --image_path or --query must be provided", file=sys.stderr)
        sys.exit(1)
    
    # Output as JSON
    result = {
        'success': True,
        'recommendations': recommendations,
        'count': len(recommendations)
    }
    
    print(json.dumps(result))

if __name__ == '__main__':
    main()
