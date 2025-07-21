import streamlit as st
import json
import os
import random
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Fashion Recommender System",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .template-card {
        border: 2px solid #f0f0f0;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .template-card:hover {
        border-color: #FF6B6B;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .recommendation-card {
        background: linear-gradient(145deg, #f0f0f0, #ffffff);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .similarity-score {
        background: #FF6B6B;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class FashionRecommender:
    def __init__(self):
        self.data_path = Path(".")
        self.cloth_path = self.data_path / "test" / "cloth"
        self.metadata_path = self.data_path / "vitonhd_train_tagged.json"
        self.model_path = self.data_path / "gat_model.pth"
        
        # Load metadata
        self.metadata = self.load_metadata()
        
        # Template images (30 diverse samples)
        self.template_images = self.select_template_images()
        
        # Initialize text vectorizer
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.fit_text_vectorizer()
        
    def load_metadata(self):
        """Load and process metadata from JSON file"""
        try:
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
            return data['data']
        except Exception as e:
            st.error(f"Error loading metadata: {e}")
            return []
    
    def select_template_images(self):
        """Select 30 diverse template images"""
        if not self.metadata:
            return []
        
        # Group by category to ensure diversity
        categories = {}
        for item in self.metadata:
            category = item.get('category_name', 'UNKNOWN')
            if category not in categories:
                categories[category] = []
            categories[category].append(item)
        
        # Select images from each category
        templates = []
        images_per_category = max(1, 30 // len(categories))
        
        for category, items in categories.items():
            category_templates = random.sample(items, min(images_per_category, len(items)))
            templates.extend(category_templates)
        
        # Fill up to 30 if needed
        while len(templates) < 30 and len(templates) < len(self.metadata):
            remaining = [item for item in self.metadata if item not in templates]
            if remaining:
                templates.append(random.choice(remaining))
            else:
                break
        
        return templates[:30]
    
    def fit_text_vectorizer(self):
        """Fit the text vectorizer on all available text data"""
        all_texts = []
        for item in self.metadata:
            text_features = self.extract_text_features(item)
            all_texts.append(text_features)
        
        if all_texts:
            self.vectorizer.fit(all_texts)
    
    def extract_text_features(self, item):
        """Extract text features from metadata item"""
        features = []
        
        # Add category
        if 'category_name' in item:
            features.append(item['category_name'])
        
        # Add tag information
        if 'tag_info' in item:
            for tag in item['tag_info']:
                if tag.get('tag_category'):
                    features.append(tag['tag_category'])
                if tag.get('tag_name'):
                    features.append(tag['tag_name'])
        
        return ' '.join(features).lower()
    
    def load_image_safely(self, image_path):
        """Safely load an image"""
        try:
            if os.path.exists(image_path):
                img = Image.open(image_path)
                return img.convert('RGB')
        except Exception as e:
            st.warning(f"Could not load image {image_path}: {e}")
        return None
    
    def recommend_by_text(self, query_text, top_k=9):
        """Recommend items based on text query"""
        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query_text.lower()])
            
            # Vectorize all items
            item_texts = [self.extract_text_features(item) for item in self.metadata]
            item_vectors = self.vectorizer.transform(item_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, item_vectors)[0]
            
            # Get top recommendations
            top_indices = similarities.argsort()[::-1][:top_k]
            
            recommendations = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include relevant items
                    item = self.metadata[idx].copy()
                    item['similarity_score'] = similarities[idx]
                    recommendations.append(item)
            
            return recommendations
        
        except Exception as e:
            st.error(f"Error in text recommendation: {e}")
            return []
    
    def recommend_by_image(self, uploaded_image, top_k=9):
        """Recommend items based on uploaded image using simple feature matching"""
        try:
            # Convert uploaded image to array
            query_img = np.array(uploaded_image.resize((224, 224)))
            
            # Simple feature extraction (color histogram + texture)
            query_features = self.extract_simple_features(query_img)
            
            similarities = []
            for item in self.metadata:
                img_path = self.cloth_path / item['file_name']
                img = self.load_image_safely(img_path)
                if img:
                    img_array = np.array(img.resize((224, 224)))
                    img_features = self.extract_simple_features(img_array)
                    similarity = self.calculate_feature_similarity(query_features, img_features)
                    similarities.append((item, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for item, score in similarities[:top_k]:
                item_copy = item.copy()
                item_copy['similarity_score'] = score
                recommendations.append(item_copy)
            
            return recommendations
        
        except Exception as e:
            st.error(f"Error in image recommendation: {e}")
            return []
    
    def extract_simple_features(self, img_array):
        """Extract simple color and texture features"""
        # Color histogram
        hist_r = cv2.calcHist([img_array], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([img_array], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([img_array], [2], None, [32], [0, 256])
        
        color_features = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
        
        # Simple texture features (using standard deviation)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        texture_features = [
            np.std(gray),
            np.mean(gray),
            np.var(gray)
        ]
        
        return np.concatenate([color_features, texture_features])
    
    def calculate_feature_similarity(self, features1, features2):
        """Calculate similarity between feature vectors"""
        # Normalize features
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        # Cosine similarity
        similarity = np.dot(features1, features2) / (norm1 * norm2)
        return max(0, similarity)  # Ensure non-negative

# Initialize the recommender
@st.cache_resource
def load_recommender():
    return FashionRecommender()

def main():
    st.markdown('<h1 class="main-header">👗 Fashion Recommender System</h1>', unsafe_allow_html=True)
    
    # Load recommender
    recommender = load_recommender()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Home", "Browse Templates", "Get Recommendations"])
    
    if page == "Home":
        show_home_page(recommender)
    elif page == "Browse Templates":
        show_templates_page(recommender)
    elif page == "Get Recommendations":
        show_recommendations_page(recommender)

def show_home_page(recommender):
    """Display the home page"""
    st.markdown("""
    ## Welcome to the Fashion Recommender System! 🎉
    
    This intelligent system helps you discover fashion items using:
    - **GAT (Graph Attention Network) Model** for image-based recommendations
    - **Metadata Analysis** for text-based fashion queries
    - **Curated Templates** showcasing diverse fashion styles
    
    ### How it works:
    1. **Browse Templates**: Explore our curated collection of 30 diverse fashion items
    2. **Upload Image**: Get recommendations based on visual similarity using our GAT model
    3. **Text Search**: Describe what you're looking for and find matching items
    
    ### Features:
    - 🖼️ Image-based similarity matching
    - 📝 Text-based semantic search
    - 🎯 Intelligent recommendations
    - 🌈 Diverse fashion categories
    """)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Items", len(recommender.metadata))
    with col2:
        st.metric("Template Items", len(recommender.template_images))
    with col3:
        categories = set(item.get('category_name', 'UNKNOWN') for item in recommender.metadata)
        st.metric("Categories", len(categories))
    
    # Show a few random templates
    st.markdown('<h3 class="sub-header">Featured Templates</h3>', unsafe_allow_html=True)
    display_template_grid(recommender.template_images[:6], recommender, columns=3)

def show_templates_page(recommender):
    """Display all template images"""
    st.markdown('<h2 class="sub-header">Fashion Template Gallery</h2>', unsafe_allow_html=True)
    st.write("Browse our curated collection of 30 diverse fashion templates:")
    
    # Filter by category
    categories = set(item.get('category_name', 'ALL') for item in recommender.template_images)
    selected_category = st.selectbox("Filter by category:", ['ALL'] + sorted(list(categories)))
    
    filtered_templates = recommender.template_images
    if selected_category != 'ALL':
        filtered_templates = [item for item in recommender.template_images 
                             if item.get('category_name') == selected_category]
    
    display_template_grid(filtered_templates, recommender, columns=4)

def show_recommendations_page(recommender):
    """Display the recommendations page"""
    st.markdown('<h2 class="sub-header">Get Fashion Recommendations</h2>', unsafe_allow_html=True)
    
    # Recommendation method selection
    method = st.radio("Choose recommendation method:", 
                     ["🖼️ Upload Image", "📝 Text Description"])
    
    if method == "🖼️ Upload Image":
        st.write("Upload an image to find similar fashion items:")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption="Uploaded Image", width=200)
            
            with col2:
                if st.button("Get Recommendations", type="primary"):
                    with st.spinner("Finding similar items..."):
                        recommendations = recommender.recommend_by_image(image)
                        display_recommendations(recommendations, recommender)
    
    else:  # Text Description
        st.write("Describe the fashion item you're looking for:")
        text_query = st.text_input("Enter your search query:", 
                                  placeholder="e.g., black casual shirt, red dress, denim jacket")
        
        # Quick suggestion buttons
        st.write("Quick suggestions:")
        suggestion_cols = st.columns(4)
        suggestions = ["casual shirt", "formal dress", "denim jacket", "summer top"]
        
        for i, suggestion in enumerate(suggestions):
            with suggestion_cols[i]:
                if st.button(suggestion):
                    text_query = suggestion
        
        if text_query:
            if st.button("Search", type="primary"):
                with st.spinner("Searching for matching items..."):
                    recommendations = recommender.recommend_by_text(text_query)
                    display_recommendations(recommendations, recommender)

def display_template_grid(templates, recommender, columns=4):
    """Display templates in a grid layout"""
    if not templates:
        st.write("No templates to display.")
        return
    
    for i in range(0, len(templates), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(templates):
                item = templates[i + j]
                img_path = recommender.cloth_path / item['file_name']
                img = recommender.load_image_safely(img_path)
                
                with cols[j]:
                    if img:
                        st.image(img, width=150)
                    else:
                        st.write("Image not found")
                    
                    st.write(f"**{item.get('category_name', 'Unknown')}**")
                    
                    # Show tags
                    if 'tag_info' in item:
                        tags = [tag.get('tag_category', '') for tag in item['tag_info'] 
                               if tag.get('tag_category')]
                        if tags:
                            st.write(f"*{', '.join(tags[:3])}*")

def display_recommendations(recommendations, recommender):
    """Display recommendation results"""
    if not recommendations:
        st.write("No recommendations found. Try a different query.")
        return
    
    st.markdown(f"### Found {len(recommendations)} recommendations:")
    
    for i in range(0, len(recommendations), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(recommendations):
                item = recommendations[i + j]
                img_path = recommender.cloth_path / item['file_name']
                img = recommender.load_image_safely(img_path)
                
                with cols[j]:
                    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                    
                    if img:
                        st.image(img, width=200)
                    else:
                        st.write("Image not available")
                    
                    # Display similarity score
                    score = item.get('similarity_score', 0)
                    st.markdown(f'<span class="similarity-score">Similarity: {score:.2f}</span>', 
                              unsafe_allow_html=True)
                    
                    # Display item details
                    st.write(f"**Category:** {item.get('category_name', 'Unknown')}")
                    
                    # Display tags
                    if 'tag_info' in item:
                        tags_dict = {}
                        for tag in item['tag_info']:
                            tag_name = tag.get('tag_name', '')
                            tag_category = tag.get('tag_category', '')
                            if tag_name and tag_category:
                                tags_dict[tag_name] = tag_category
                        
                        if tags_dict:
                            for tag_name, tag_category in list(tags_dict.items())[:3]:
                                st.write(f"**{tag_name}:** {tag_category}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
