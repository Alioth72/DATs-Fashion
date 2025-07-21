#!/usr/bin/env python3
"""
Text Recommender Script for Next.js API
Provides text-based fashion recommendations using metadata
"""

import sys
import json
import argparse
from pathlib import Path
import re
from collections import defaultdict
import random

def normalize_text(text):
    """Normalize text for comparison"""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower().strip())

def extract_keywords(query):
    """Extract keywords from query"""
    normalized = normalize_text(query)
    # Common fashion keywords and patterns
    keywords = normalized.split()
    return [kw for kw in keywords if len(kw) > 2]

def calculate_text_similarity(query_keywords, item_text):
    """Calculate similarity score between query and item"""
    item_text = normalize_text(item_text)
    item_words = set(item_text.split())
    
    score = 0
    for keyword in query_keywords:
        if keyword in item_text:
            score += 2  # Exact match
        else:
            # Partial match
            for word in item_words:
                if keyword in word or word in keyword:
                    score += 0.5
    
    return score / max(len(query_keywords), 1)

def load_metadata(metadata_path):
    """Load metadata from JSON file"""
    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        return data['data']
    except Exception as e:
        print(f"Error loading metadata: {e}", file=sys.stderr)
        return []

def get_text_recommendations(query, metadata_path, cloth_dir, top_k=12):
    """Get recommendations based on text query"""
    try:
        # Load metadata
        metadata = load_metadata(metadata_path)
        if not metadata:
            return []
        
        # Extract keywords from query
        query_keywords = extract_keywords(query)
        if not query_keywords:
            # Return random items if no keywords
            available_items = []
            cloth_path = Path(cloth_dir)
            
            for item in metadata[:100]:  # Sample first 100
                img_path = cloth_path / item['file_name']
                if img_path.exists():
                    item_copy = item.copy()
                    item_copy['image_url'] = f"/api/images/{item['file_name']}"
                    item_copy['similarity_score'] = random.uniform(0.3, 0.8)
                    available_items.append(item_copy)
            
            return random.sample(available_items, min(top_k, len(available_items)))
        
        # Score all items
        scored_items = []
        cloth_path = Path(cloth_dir)
        
        for item in metadata:
            img_path = cloth_path / item['file_name']
            if not img_path.exists():
                continue
                
            # Combine all text fields for matching
            text_fields = []
            
            # Add category information
            if 'category_name' in item:
                text_fields.append(item['category_name'])
            if 'sub_category_name' in item:
                text_fields.append(item['sub_category_name'])
                
            # Add product details
            if 'product_display_name' in item:
                text_fields.append(item['product_display_name'])
                
            # Add tags if available
            if 'tags' in item:
                if isinstance(item['tags'], list):
                    text_fields.extend(item['tags'])
                else:
                    text_fields.append(str(item['tags']))
            
            # Add color and other attributes
            for field in ['base_colour', 'season', 'usage', 'product_type']:
                if field in item and item[field]:
                    text_fields.append(str(item[field]))
            
            # Calculate similarity
            combined_text = ' '.join(text_fields)
            similarity = calculate_text_similarity(query_keywords, combined_text)
            
            if similarity > 0:  # Only include items with some relevance
                item_copy = item.copy()
                item_copy['similarity_score'] = similarity
                item_copy['image_url'] = f"/api/images/{item['file_name']}"
                scored_items.append(item_copy)
        
        # Sort by similarity and return top_k
        scored_items.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # If we don't have enough high-scoring items, add some random ones
        if len(scored_items) < top_k:
            remaining_items = []
            for item in metadata:
                img_path = cloth_path / item['file_name']
                if img_path.exists() and not any(si['file_name'] == item['file_name'] for si in scored_items):
                    item_copy = item.copy()
                    item_copy['similarity_score'] = random.uniform(0.1, 0.3)
                    item_copy['image_url'] = f"/api/images/{item['file_name']}"
                    remaining_items.append(item_copy)
            
            additional_needed = top_k - len(scored_items)
            scored_items.extend(random.sample(remaining_items, min(additional_needed, len(remaining_items))))
        
        return scored_items[:top_k]
        
    except Exception as e:
        print(f"Error in get_text_recommendations: {e}", file=sys.stderr)
        return []

def main():
    parser = argparse.ArgumentParser(description='Text-based Fashion Recommender')
    parser.add_argument('--query', required=True, help='Search query')
    parser.add_argument('--metadata_path', required=True, help='Path to metadata JSON')
    parser.add_argument('--cloth_dir', required=True, help='Path to cloth images directory')
    parser.add_argument('--top_k', type=int, default=12, help='Number of recommendations')
    
    args = parser.parse_args()
    
    # Get recommendations
    recommendations = get_text_recommendations(
        args.query,
        args.metadata_path,
        args.cloth_dir,
        args.top_k
    )
    
    # Output as JSON
    result = {
        'success': True,
        'recommendations': recommendations,
        'count': len(recommendations),
        'query': args.query
    }
    
    print(json.dumps(result))

if __name__ == '__main__':
    main()
