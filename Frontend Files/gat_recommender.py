#!/usr/bin/env python3
"""
GAT Recommender Script for Next.js API
Interfaces with the GAT model to provide image-based recommendations
"""

import sys
import json
import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

try:
    from model1_architecture import FashionGATModel
    from enhanced_gat_integration import EnhancedFashionGATRecommender
except ImportError as e:
    print(f"Error importing modules: {e}", file=sys.stderr)
    sys.exit(1)

def load_metadata(metadata_path):
    """Load metadata from JSON file"""
    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        return data['data']
    except Exception as e:
        print(f"Error loading metadata: {e}", file=sys.stderr)
        return []

def get_image_recommendations(image_path, model_path, metadata_path, cloth_dir, top_k=9):
    """Get recommendations for an uploaded image"""
    try:
        # Initialize the GAT recommender
        recommender = EnhancedFashionGATRecommender(model_path)
        
        # Load the uploaded image
        uploaded_image = Image.open(image_path).convert('RGB')
        
        # Load metadata
        metadata = load_metadata(metadata_path)
        if not metadata:
            return []
        
        # Get available items (those with existing images)
        available_items = []
        candidate_images = []
        cloth_path = Path(cloth_dir)
        
        # Sample a subset for faster processing (first 500 items)
        sample_size = min(500, len(metadata))
        sample_metadata = metadata[:sample_size]
        
        for item in sample_metadata:
            img_path = cloth_path / item['file_name']
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    available_items.append(item)
                    candidate_images.append(img)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}", file=sys.stderr)
                    continue
        
        if not candidate_images:
            return []
        
        # Use GAT model for recommendations
        similar_indices = recommender.recommend_by_image(
            uploaded_image, candidate_images, top_k
        )
        
        recommendations = []
        for idx, score in similar_indices:
            if idx < len(available_items):
                item = available_items[idx].copy()
                item['similarity_score'] = float(score)
                # Add image URL for frontend
                item['image_url'] = f"/api/images/{item['file_name']}"
                recommendations.append(item)
        
        return recommendations
        
    except Exception as e:
        print(f"Error in get_image_recommendations: {e}", file=sys.stderr)
        return []

def main():
    parser = argparse.ArgumentParser(description='GAT Image Recommender')
    parser.add_argument('--image_path', required=True, help='Path to uploaded image')
    parser.add_argument('--model_path', required=True, help='Path to GAT model')
    parser.add_argument('--metadata_path', required=True, help='Path to metadata JSON')
    parser.add_argument('--cloth_dir', required=True, help='Path to cloth images directory')
    parser.add_argument('--top_k', type=int, default=9, help='Number of recommendations')
    
    args = parser.parse_args()
    
    # Get recommendations
    recommendations = get_image_recommendations(
        args.image_path,
        args.model_path, 
        args.metadata_path,
        args.cloth_dir,
        args.top_k
    )
    
    # Output as JSON
    result = {
        'success': True,
        'recommendations': recommendations,
        'count': len(recommendations)
    }
    
    print(json.dumps(result))

if __name__ == '__main__':
    main()
