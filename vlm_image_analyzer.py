#!/usr/bin/env python3
"""
Vision-Language Model for Image Analysis
Uses a lightweight VLM to analyze fashion images and generate descriptive text
"""

import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings
warnings.filterwarnings("ignore")

class FashionImageAnalyzer:
    """Analyzes fashion images using Vision-Language Model"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[VLM] Using device: {self.device}", file=sys.stderr)
        
        try:
            # Use BLIP model for image captioning (lightweight alternative to LLaVA)
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model.to(self.device)
            self.model_loaded = True
            print("[VLM] BLIP model loaded successfully", file=sys.stderr)
        except Exception as e:
            print(f"[VLM] Error loading BLIP model: {e}", file=sys.stderr)
            self.model_loaded = False
    
    def analyze_fashion_image(self, image_path, detailed=True):
        """
        Analyze a fashion image and extract descriptive features
        
        Args:
            image_path: Path to the image file
            detailed: Whether to generate detailed descriptions
            
        Returns:
            dict: Analysis results with description and extracted features
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            if self.model_loaded:
                return self._analyze_with_blip(image, detailed)
            else:
                return self._analyze_with_fallback(image)
                
        except Exception as e:
            print(f"[VLM] Error analyzing image: {e}", file=sys.stderr)
            return self._get_default_analysis()
    
    def _analyze_with_blip(self, image, detailed=True):
        """Analyze image using BLIP model"""
        try:
            # Generate basic caption
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_length=50, num_beams=5)
            basic_caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Generate detailed fashion-specific description
            if detailed:
                # Use conditional text generation for fashion-specific prompts
                conditional_inputs = self.processor(
                    image, 
                    text="This fashion item is", 
                    return_tensors="pt"
                ).to(self.device)
                
                fashion_out = self.model.generate(
                    **conditional_inputs, 
                    max_length=100, 
                    num_beams=5,
                    temperature=0.7
                )
                fashion_description = self.processor.decode(fashion_out[0], skip_special_tokens=True)
                
                # Extract features from descriptions
                features = self._extract_features_from_text(basic_caption + " " + fashion_description)
            else:
                fashion_description = basic_caption
                features = self._extract_features_from_text(basic_caption)
            
            return {
                "success": True,
                "basic_caption": basic_caption,
                "fashion_description": fashion_description,
                "extracted_features": features,
                "analysis_method": "BLIP"
            }
            
        except Exception as e:
            print(f"[VLM] Error in BLIP analysis: {e}", file=sys.stderr)
            return self._get_default_analysis()
    
    def _analyze_with_fallback(self, image):
        """Fallback analysis without VLM"""
        return {
            "success": True,
            "basic_caption": "fashion clothing item",
            "fashion_description": "A stylish clothing item suitable for various occasions",
            "extracted_features": {
                "item_type": "clothing",
                "color": "unknown",
                "style": "casual",
                "material": "fabric",
                "season": "all-season"
            },
            "analysis_method": "fallback"
        }
    
    def _extract_features_from_text(self, description):
        """Extract structured features from text description"""
        description_lower = description.lower()
        
        # Fashion item types
        item_types = ["shirt", "dress", "pants", "skirt", "jacket", "blouse", "sweater", "jeans", "shorts", "top"]
        detected_item = next((item for item in item_types if item in description_lower), "clothing")
        
        # Colors
        colors = ["black", "white", "blue", "red", "green", "yellow", "pink", "purple", "gray", "brown", "orange"]
        detected_color = next((color for color in colors if color in description_lower), "unknown")
        
        # Styles
        styles = ["casual", "formal", "elegant", "sporty", "vintage", "modern", "classic", "trendy"]
        detected_style = next((style for style in styles if style in description_lower), "casual")
        
        # Materials
        materials = ["cotton", "silk", "wool", "denim", "leather", "polyester", "linen", "fabric"]
        detected_material = next((material for material in materials if material in description_lower), "fabric")
        
        # Occasions/Usage
        occasions = ["casual", "formal", "party", "work", "summer", "winter", "spring", "fall"]
        detected_occasion = next((occasion for occasion in occasions if occasion in description_lower), "casual")
        
        return {
            "item_type": detected_item,
            "color": detected_color,
            "style": detected_style,
            "material": detected_material,
            "occasion": detected_occasion,
            "description": description
        }
    
    def _get_default_analysis(self):
        """Default analysis when all else fails"""
        return {
            "success": False,
            "basic_caption": "fashion item",
            "fashion_description": "A clothing item",
            "extracted_features": {
                "item_type": "clothing",
                "color": "unknown",
                "style": "casual",
                "material": "fabric",
                "occasion": "casual"
            },
            "analysis_method": "default"
        }

def main():
    parser = argparse.ArgumentParser(description='Fashion Image Analyzer using VLM')
    parser.add_argument('--image_path', required=True, help='Path to image file')
    parser.add_argument('--detailed', action='store_true', help='Generate detailed analysis')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FashionImageAnalyzer()
    
    # Analyze image
    result = analyzer.analyze_fashion_image(args.image_path, args.detailed)
    
    # Output JSON
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
