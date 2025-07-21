"""
LLM Query Enhancer for Fashion Search
Uses Google Gemini AI to enhance user queries for better fashion recommendations
"""

import google.generativeai as genai
import json
import logging
from typing import Dict, Optional

class FashionQueryEnhancer:
    def __init__(self, api_key: str):
        """Initialize the query enhancer with Google Gemini API"""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def enhance_query(self, user_query: str, detailed: bool = False) -> Dict:
        """
        Enhance a user's fashion query using LLM
        
        Args:
            user_query: The original user query
            detailed: Whether to provide detailed analysis
            
        Returns:
            Dictionary containing enhanced query and analysis
        """
        try:
            # Create the enhancement prompt
            prompt = self._create_enhancement_prompt(user_query, detailed)
            
            # Generate enhanced query
            response = self.model.generate_content(prompt)
            
            # Parse the response
            enhanced_data = self._parse_response(response.text, user_query)
            
            self.logger.info(f"Enhanced query: {user_query} -> {enhanced_data['enhanced_query']}")
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"Error enhancing query: {e}")
            return {
                'success': False,
                'error': str(e),
                'original_query': user_query,
                'enhanced_query': user_query,  # Fallback to original
                'features': {},
                'suggestions': []
            }
    
    def _create_enhancement_prompt(self, user_query: str, detailed: bool) -> str:
        """Create the prompt for query enhancement"""
        
        base_prompt = f"""
You are a fashion expert AI assistant. Enhance the following fashion search query to make it more specific and searchable.

Original Query: "{user_query}"

Please provide a comprehensive enhanced query that includes:
1. Specific clothing item type
2. Colors and patterns
3. Style and aesthetic
4. Material/fabric details
5. Occasion or use case
6. Fit and silhouette

Transform vague terms into specific fashion terminology. For example:
- "nice shirt" -> "casual cotton button-down shirt"
- "party dress" -> "elegant evening cocktail dress"
- "comfortable shoes" -> "casual leather sneakers"

Response Format (JSON):
{{
    "enhanced_query": "detailed enhanced search query (15-20 words)",
    "item_type": "specific clothing category",
    "colors": ["color1", "color2"],
    "style": "style category (casual, formal, vintage, etc.)",
    "material": "fabric/material type",
    "occasion": "when/where to wear",
    "fit": "fit type (slim, regular, oversized, etc.)",
    "features": ["feature1", "feature2", "feature3"],
    "search_tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}}
"""
        
        if detailed:
            base_prompt += """
            
Additionally provide:
- "alternatives": ["alternative search term 1", "alternative search term 2"]
- "styling_tips": "how to style this item"
- "season": "best season for this item"
- "price_range": "budget category (budget/mid-range/premium)"
"""
        
        return base_prompt
    
    def _parse_response(self, response_text: str, original_query: str) -> Dict:
        """Parse the LLM response into structured data"""
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                parsed_data = json.loads(json_str)
                
                # Ensure required fields exist
                result = {
                    'success': True,
                    'original_query': original_query,
                    'enhanced_query': parsed_data.get('enhanced_query', original_query),
                    'item_type': parsed_data.get('item_type', ''),
                    'colors': parsed_data.get('colors', []),
                    'style': parsed_data.get('style', ''),
                    'material': parsed_data.get('material', ''),
                    'occasion': parsed_data.get('occasion', ''),
                    'fit': parsed_data.get('fit', ''),
                    'features': parsed_data.get('features', []),
                    'search_tags': parsed_data.get('search_tags', []),
                    'alternatives': parsed_data.get('alternatives', []),
                    'styling_tips': parsed_data.get('styling_tips', ''),
                    'season': parsed_data.get('season', ''),
                    'price_range': parsed_data.get('price_range', '')
                }
                
                return result
            
            else:
                # Fallback: create basic enhancement
                return self._create_fallback_enhancement(response_text, original_query)
                
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return self._create_fallback_enhancement(response_text, original_query)
    
    def _create_fallback_enhancement(self, response_text: str, original_query: str) -> Dict:
        """Create a fallback enhancement if JSON parsing fails"""
        # Use the response text as enhanced query, or fallback to original
        enhanced_query = response_text.strip() if response_text.strip() else original_query
        
        # Remove any JSON artifacts
        enhanced_query = enhanced_query.replace('{', '').replace('}', '').replace('"', '')
        
        # Limit length
        if len(enhanced_query) > 100:
            enhanced_query = enhanced_query[:100] + "..."
        
        return {
            'success': True,
            'original_query': original_query,
            'enhanced_query': enhanced_query,
            'item_type': '',
            'colors': [],
            'style': '',
            'material': '',
            'occasion': '',
            'fit': '',
            'features': [],
            'search_tags': [],
            'alternatives': [],
            'styling_tips': '',
            'season': '',
            'price_range': ''
        }
    
    def get_search_suggestions(self, category: str = "general") -> list:
        """Get search suggestions for different categories"""
        suggestions = {
            "general": [
                "casual weekend outfit",
                "formal business attire",
                "summer vacation wear",
                "cozy winter clothing",
                "party evening dress"
            ],
            "tops": [
                "oversized cotton hoodie",
                "silk button-down blouse",
                "vintage band t-shirt",
                "elegant off-shoulder top",
                "athletic moisture-wicking shirt"
            ],
            "bottoms": [
                "high-waisted skinny jeans",
                "flowy maxi skirt",
                "tailored dress pants",
                "denim boyfriend shorts",
                "pleated tennis skirt"
            ],
            "dresses": [
                "little black cocktail dress",
                "bohemian maxi sundress",
                "professional midi dress",
                "vintage swing dress",
                "elegant evening gown"
            ],
            "outerwear": [
                "cozy oversized cardigan",
                "classic trench coat",
                "denim jacket vintage",
                "leather motorcycle jacket",
                "wool winter peacoat"
            ]
        }
        
        return suggestions.get(category, suggestions["general"])

# Test function
def test_query_enhancer():
    """Test the query enhancer with sample queries"""
    api_key = "AIzaSyAHnaMqSqZzwZaoGy0TjOhrWJZFuM6Wg9k"  # Your API key
    enhancer = FashionQueryEnhancer(api_key)
    
    test_queries = [
        "nice shirt",
        "party dress", 
        "comfortable shoes",
        "warm jacket",
        "summer outfit"
    ]
    
    for query in test_queries:
        print(f"\nTesting: {query}")
        result = enhancer.enhance_query(query, detailed=True)
        if result['success']:
            print(f"Enhanced: {result['enhanced_query']}")
            print(f"Features: {result['features']}")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    test_query_enhancer()
