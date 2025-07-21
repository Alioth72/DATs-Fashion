#!/usr/bin/env python3
"""
Enhanced GAT Recommender with VLM Integration
Combines Vision-Language Model analysis with GAT model for better recommendations
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

try:
    from model1_architecture import FashionGATModel
    from enhanced_gat_integration import EnhancedFashionGATRecommender
    from vlm_image_analyzer import FashionImageAnalyzer
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

def get_enhanced_image_recommendations(image_path, model_path, metadata_path, cloth_dir, top_k=9):
    """
    Get recommendations for an uploaded image using VLM + GAT model
    
    This approach:
    1. Uses VLM to analyze the uploaded image and extract semantic features
    2. Creates a text query based on the image analysis  
    3. Uses the GAT model with the generated text query for recommendations
    4. Also uses visual similarity as a secondary ranking factor
    """
    try:
        print("[INFO] Starting enhanced image recommendation process", file=sys.stderr)
        
        # Step 1: Analyze image with VLM
        analyzer = FashionImageAnalyzer()
        image_analysis = analyzer.analyze_fashion_image(image_path, detailed=True)
        
        if not image_analysis['success']:
            print("[WARNING] VLM analysis failed, using fallback method", file=sys.stderr)
            return get_fallback_recommendations(image_path, model_path, metadata_path, cloth_dir, top_k)
        
        print(f"[INFO] VLM Analysis: {image_analysis['fashion_description']}", file=sys.stderr)
        
        # Step 2: Create text query from image analysis
        features = image_analysis['extracted_features']
        text_query = f"{features['item_type']} {features['color']} {features['style']} {features['material']} {features['occasion']}"
        
        print(f"[INFO] Generated query: {text_query}", file=sys.stderr)
        
        # Step 3: Use GAT model with text query
        text_recommendations = get_text_based_recommendations(
            text_query, metadata_path, cloth_dir, top_k * 2  # Get more for reranking
        )
        
        # Step 4: Enhance with visual similarity
        visual_recommendations = get_visual_similarity_recommendations(
            image_path, metadata_path, cloth_dir, top_k
        )
        
        # Step 5: Combine and rerank recommendations
        final_recommendations = combine_recommendations(
            text_recommendations, visual_recommendations, image_analysis, top_k
        )
        
        print(f"[INFO] Returning {len(final_recommendations)} recommendations", file=sys.stderr)
        return final_recommendations
        
    except Exception as e:
        print(f"Error in enhanced image recommendations: {e}", file=sys.stderr)
        return get_fallback_recommendations(image_path, model_path, metadata_path, cloth_dir, top_k)

def get_text_based_recommendations(query, metadata_path, cloth_dir, top_k):
    """Get recommendations using text-based GAT model"""
    try:
        # Initialize GAT recommender
        model = FashionGATModel()
        metadata = load_metadata(metadata_path)
        
        if not metadata:
            return []
        
        # Use GAT model for text-based recommendation
        similar_items = model.find_similar_items_by_text(query, metadata, top_k)
        
        recommendations = []
        cloth_path = Path(cloth_dir)
        
        for idx, score in similar_items:
            if idx < len(metadata):
                item = metadata[idx].copy()
                item['similarity_score'] = float(score)
                item['recommendation_type'] = 'text_based'
                
                # Check if image exists
                img_path = cloth_path / item['file_name']
                if img_path.exists():
                    item['image_url'] = f"/api/images/{item['file_name']}"
                    recommendations.append(item)
        
        return recommendations
        
    except Exception as e:
        print(f"Error in text-based recommendations: {e}", file=sys.stderr)
        return []

def get_visual_similarity_recommendations(image_path, metadata_path, cloth_dir, top_k):
    """Get recommendations using visual similarity"""
    try:
        # Initialize visual recommender
        recommender = EnhancedFashionGATRecommender()
        
        # Load uploaded image
        uploaded_image = Image.open(image_path).convert('RGB')
        
        # Load metadata and candidate images
        metadata = load_metadata(metadata_path)
        if not metadata:
            return []
        
        available_items = []
        candidate_images = []
        cloth_path = Path(cloth_dir)
        
        # Sample subset for faster processing
        sample_size = min(200, len(metadata))
        sample_metadata = metadata[:sample_size]
        
        for item in sample_metadata:
            img_path = cloth_path / item['file_name']
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    available_items.append(item)
                    candidate_images.append(img)
                except Exception:
                    continue
        
        if not candidate_images:
            return []
        
        # Get visual similarities
        similar_indices = recommender.recommend_by_image(
            uploaded_image, candidate_images, top_k
        )
        
        recommendations = []
        for idx, score in similar_indices:
            if idx < len(available_items):
                item = available_items[idx].copy()
                item['similarity_score'] = float(score)
                item['recommendation_type'] = 'visual_similarity'
                item['image_url'] = f"/api/images/{item['file_name']}"
                recommendations.append(item)
        
        return recommendations
        
    except Exception as e:
        print(f"Error in visual similarity recommendations: {e}", file=sys.stderr)
        return []

def combine_recommendations(text_recs, visual_recs, image_analysis, top_k):
    """
    Combine text-based and visual recommendations intelligently
    """
    try:
        # Create a scoring system
        combined_items = {}
        features = image_analysis['extracted_features']
        
        # Add text-based recommendations (higher weight)
        for item in text_recs:
            file_name = item['file_name']
            combined_items[file_name] = {
                'item': item,
                'text_score': item['similarity_score'],
                'visual_score': 0.0,
                'combined_score': item['similarity_score'] * 0.7  # 70% weight for text
            }
        
        # Add visual recommendations
        for item in visual_recs:
            file_name = item['file_name']
            if file_name in combined_items:
                # Item found in both - boost score
                combined_items[file_name]['visual_score'] = item['similarity_score']
                combined_items[file_name]['combined_score'] += item['similarity_score'] * 0.3  # 30% weight for visual
                combined_items[file_name]['combined_score'] *= 1.2  # Boost for appearing in both
            else:
                # Visual-only recommendation
                combined_items[file_name] = {
                    'item': item,
                    'text_score': 0.0,
                    'visual_score': item['similarity_score'],
                    'combined_score': item['similarity_score'] * 0.5  # Lower weight for visual-only
                }
        
        # Apply semantic filtering based on VLM analysis
        for file_name, data in combined_items.items():
            item = data['item']
            
            # Boost score if item matches VLM-detected features
            if 'tag_info' in item:
                for tag in item.get('tag_info', []):
                    tag_category = tag.get('tag_category', '').lower()
                    tag_name = tag.get('tag_name', '').lower()
                    
                    # Check matches with VLM features
                    if (features['item_type'] in tag_category or 
                        features['color'] in tag_category or
                        features['style'] in tag_category):
                        data['combined_score'] *= 1.1  # Small boost for semantic match
        
        # Sort by combined score and return top_k
        sorted_items = sorted(combined_items.values(), key=lambda x: x['combined_score'], reverse=True)
        
        final_recommendations = []
        for data in sorted_items[:top_k]:
            item = data['item'].copy()
            item['final_score'] = data['combined_score']
            item['text_component'] = data['text_score']
            item['visual_component'] = data['visual_score']
            final_recommendations.append(item)
        
        return final_recommendations
        
    except Exception as e:
        print(f"Error combining recommendations: {e}", file=sys.stderr)
        # Fallback to text recommendations
        return text_recs[:top_k] if text_recs else visual_recs[:top_k]

def get_fallback_recommendations(image_path, model_path, metadata_path, cloth_dir, top_k):
    """Fallback to original GAT recommender if VLM fails"""
    try:
        recommender = EnhancedFashionGATRecommender(model_path)
        uploaded_image = Image.open(image_path).convert('RGB')
        
        metadata = load_metadata(metadata_path)
        if not metadata:
            return []
        
        available_items = []
        candidate_images = []
        cloth_path = Path(cloth_dir)
        
        sample_size = min(500, len(metadata))
        for item in metadata[:sample_size]:
            img_path = cloth_path / item['file_name']
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    available_items.append(item)
                    candidate_images.append(img)
                except Exception:
                    continue
        
        if not candidate_images:
            return []
        
        similar_indices = recommender.recommend_by_image(uploaded_image, candidate_images, top_k)
        
        recommendations = []
        for idx, score in similar_indices:
            if idx < len(available_items):
                item = available_items[idx].copy()
                item['similarity_score'] = float(score)
                item['image_url'] = f"/api/images/{item['file_name']}"
                recommendations.append(item)
        
        return recommendations
        
    except Exception as e:
        print(f"Error in fallback recommendations: {e}", file=sys.stderr)
        return []

def main():
    parser = argparse.ArgumentParser(description='Enhanced GAT Image Recommender with VLM')
    parser.add_argument('--image_path', required=True, help='Path to uploaded image')
    parser.add_argument('--model_path', required=True, help='Path to GAT model')
    parser.add_argument('--metadata_path', required=True, help='Path to metadata JSON')
    parser.add_argument('--cloth_dir', required=True, help='Path to cloth images directory')
    parser.add_argument('--top_k', type=int, default=9, help='Number of recommendations')
    
    args = parser.parse_args()
    
    # Get enhanced recommendations
    recommendations = get_enhanced_image_recommendations(
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
        'count': len(recommendations),
        'method': 'enhanced_vlm_gat'
    }
    
    print(json.dumps(result))

if __name__ == '__main__':
    main()
