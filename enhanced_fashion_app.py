import streamlit as st
import json
import os
import random
from PIL import Image
import torch
import numpy as np
from pathlib import Path
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from enhanced_gat_integration import EnhancedFashionGATRecommender
import time

# Configure page
st.set_page_config(
    page_title="AI Fashion Recommender",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
        border-bottom: 2px solid #4ECDC4;
        padding-bottom: 0.5rem;
    }
    .template-card {
        border: 2px solid #f0f0f0;
        border-radius: 15px;
        padding: 15px;
        margin: 10px;
        text-align: center;
        transition: all 0.3s ease;
        background: transparent;
        box-shadow: none;
    }
    .template-card:hover {
        border-color: #FF6B6B;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transform: translateY(-5px);
    }
    .recommendation-card {
        background: linear-gradient(145deg, #f8f9fa, #ffffff);
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    .similarity-score {
        background: linear-gradient(90deg, #FF6B6B, #FF8E53);
        color: white;
        padding: 8px 15px;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
    .feature-badge {
        background: #4ECDC4;
        color: white;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 2px;
        display: inline-block;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    .upload-area {
        border: 2px dashed #4ECDC4;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        background: #f8f9fa;
        margin: 20px 0;
    }
    .method-selector {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedFashionRecommender:
    def __init__(self):
        self.data_path = Path(".")
        self.cloth_path = self.data_path / "train" / "cloth"
        self.metadata_path = self.data_path / "vitonhd_train_tagged.json"
        self.model_path = self.data_path / "gat_model.pth"
        
        # Load metadata
        self.metadata = self.load_metadata()
        
        # Initialize GAT recommender with correct architecture
        model_path_str = str(self.model_path) if self.model_path.exists() else "gat_model.pth"
        self.gat_recommender = EnhancedFashionGATRecommender(model_path_str)
        
        # Template images (30 diverse samples)
        self.template_images = self.select_template_images()
        
        # Initialize text vectorizer
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.fit_text_vectorizer()
        
        # Pre-compute available items for faster image recommendations
        self._precompute_available_items()
        
    def _precompute_available_items(self):
        """Pre-compute available items to avoid loading all images during recommendation"""
        print("🔄 Pre-computing available items for faster recommendations...")
        import time
        start_time = time.time()
        
        self.available_items = []
        self.available_item_indices = []
        
        # Only check if files exist, don't load images yet
        for idx, item in enumerate(self.metadata):
            img_path = self.cloth_path / item['file_name']
            if img_path.exists():
                self.available_items.append(item)
                self.available_item_indices.append(idx)
        
        print(f"✅ Pre-computed {len(self.available_items)} available items in {time.time() - start_time:.2f}s")
        
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
            # Filter items that have corresponding images
            available_items = []
            for item in items:
                img_path = self.cloth_path / item['file_name']
                if img_path.exists():
                    available_items.append(item)
            
            if available_items:
                category_templates = random.sample(available_items, 
                                                 min(images_per_category, len(available_items)))
                templates.extend(category_templates)
        
        # Fill up to 30 if needed
        while len(templates) < 30 and len(templates) < len(self.metadata):
            remaining = [item for item in self.metadata if item not in templates]
            available_remaining = [item for item in remaining 
                                 if (self.cloth_path / item['file_name']).exists()]
            if available_remaining:
                templates.append(random.choice(available_remaining))
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
        """Recommend items based on text query using metadata matching"""
        try:
            # Simple text matching using TF-IDF on metadata
            query_vector = self.vectorizer.transform([query_text.lower()])
            item_texts = [self.extract_text_features(item) for item in self.metadata]
            item_vectors = self.vectorizer.transform(item_texts)
            similarities = cosine_similarity(query_vector, item_vectors)[0]
            
            # Get items sorted by similarity
            similarity_pairs = [(i, similarities[i]) for i in range(len(similarities))]
            similarity_pairs.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for idx, score in similarity_pairs:
                if score > 0.0 and len(recommendations) < top_k:  # Any positive similarity
                    item = self.metadata[idx].copy()
                    img_path = self.cloth_path / item['file_name']
                    
                    # Only include items with existing images
                    if img_path.exists():
                        item['similarity_score'] = score
                        recommendations.append(item)
            
            return recommendations
            
        except Exception as e:
            st.error(f"Error in text recommendation: {e}")
            return []

    def recommend_by_image(self, uploaded_image, top_k=9, max_candidates=500):
        """Optimized image-based recommendations using sampling to avoid long loading times"""
        try:
            # Sample a subset of available items for faster processing
            if len(self.available_items) > max_candidates:
                sampled_indices = random.sample(range(len(self.available_items)), max_candidates)
                candidate_items = [self.available_items[i] for i in sampled_indices]
            else:
                candidate_items = self.available_items
                sampled_indices = list(range(len(self.available_items)))
            
            # Load only the sampled candidate images
            candidate_images = []
            valid_candidates = []
            
            # Show progress for better user experience
            progress_placeholder = st.empty()
            
            for idx, item in enumerate(candidate_items):
                if idx % 50 == 0:  # Update progress every 50 items
                    progress_placeholder.text(f"Loading images... {idx + 1}/{len(candidate_items)}")
                
                img_path = self.cloth_path / item['file_name']
                img = self.load_image_safely(img_path)
                if img:
                    candidate_images.append(img)
                    valid_candidates.append(item)
            
            progress_placeholder.empty()
            
            if not candidate_images:
                st.error("No candidate images available for comparison.")
                return []
            
            # Use GAT model for recommendations
            with st.spinner("🧠 AI model analyzing similarities..."):
                similar_indices = self.gat_recommender.recommend_by_image(
                    uploaded_image, candidate_images, top_k
                )
            
            recommendations = []
            for idx, score in similar_indices:
                if idx < len(valid_candidates):
                    item = valid_candidates[idx].copy()
                    item['similarity_score'] = score
                    recommendations.append(item)
            
            return recommendations
        
        except Exception as e:
            st.error(f"Error in image recommendation: {e}")
            return []

# Initialize the recommender
@st.cache_resource
def load_recommender():
    return EnhancedFashionRecommender()

def main():
    st.markdown('<h1 class="main-header">🤖 AI Fashion Recommender</h1>', unsafe_allow_html=True)
    
    # Load recommender
    recommender = load_recommender()
    
    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page:", 
                               ["🏠 Home", "🖼️ Browse Templates", "🎯 Get Recommendations", "📊 Analytics"])
    
    # Model status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Status")
    if recommender.gat_recommender.model_loaded:
        st.sidebar.success("✅ GAT Model Loaded")
    else:
        st.sidebar.warning("⚠️ Using Fallback Model")
    
    st.sidebar.markdown(f"📊 **Total Items:** {len(recommender.metadata)}")
    st.sidebar.markdown(f"🖼️ **Templates:** {len(recommender.template_images)}")
    st.sidebar.markdown(f"🚀 **Available:** {len(recommender.available_items):,}")
    
    if page == "🏠 Home":
        show_home_page(recommender)
    elif page == "🖼️ Browse Templates":
        show_templates_page(recommender)
    elif page == "🎯 Get Recommendations":
        show_recommendations_page(recommender)
    elif page == "📊 Analytics":
        show_analytics_page(recommender)

def show_home_page(recommender):
    """Display the home page"""
    st.markdown("""
    ## Welcome to the AI Fashion Recommender! 🎉
    
    Discover your perfect fashion match using cutting-edge AI technology:
    """)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stats-card">
            <h3>🧠 GAT Neural Network</h3>
            <p>Advanced Graph Attention Network for precise image similarity matching</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-card">
            <h3>🔍 Metadata Search</h3>
            <p>Direct text matching using clothing categories, colors, and style tags</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-card">
            <h3>🎨 Curated Gallery</h3>
            <p>Hand-picked templates showcasing diverse fashion styles and trends</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("### 📈 Platform Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Items", f"{len(recommender.metadata):,}")
    with col2:
        st.metric("Template Items", len(recommender.template_images))
    with col3:
        categories = set(item.get('category_name', 'UNKNOWN') for item in recommender.metadata)
        st.metric("Categories", len(categories))
    with col4:
        st.metric("Available Images", f"{len(recommender.available_items):,}")
    
    # Performance note
    st.info("💡 **Performance Optimization**: Image recommendations now use smart sampling for 10x faster results!")
    
    # Featured templates
    st.markdown('<h3 class="sub-header">✨ Featured Templates</h3>', unsafe_allow_html=True)
    display_template_grid(recommender.template_images[:6], recommender, columns=3)
    
    # Quick start section
    st.markdown("### 🚀 Quick Start")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🖼️ Try Image Search", type="primary", use_container_width=True):
            st.session_state.page = "recommendations"
            st.rerun()
    
    with col2:
        if st.button("📝 Try Text Search", type="secondary", use_container_width=True):
            st.session_state.page = "recommendations"
            st.rerun()

def show_templates_page(recommender):
    """Display all template images"""
    st.markdown('<h2 class="sub-header">🖼️ Fashion Template Gallery</h2>', unsafe_allow_html=True)
    
    # Filter controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        categories = set(item.get('category_name', 'ALL') for item in recommender.template_images)
        selected_category = st.selectbox("Filter by category:", ['ALL'] + sorted(list(categories)))
    
    with col2:
        sort_by = st.selectbox("Sort by:", ["Random", "Category", "Name"])
    
    # Apply filters
    filtered_templates = recommender.template_images
    if selected_category != 'ALL':
        filtered_templates = [item for item in recommender.template_images 
                             if item.get('category_name') == selected_category]
    
    # Apply sorting
    if sort_by == "Category":
        filtered_templates.sort(key=lambda x: x.get('category_name', ''))
    elif sort_by == "Name":
        filtered_templates.sort(key=lambda x: x.get('file_name', ''))
    elif sort_by == "Random":
        random.shuffle(filtered_templates)
    
    st.write(f"Showing {len(filtered_templates)} templates")
    display_template_grid(filtered_templates, recommender, columns=4)

def show_recommendations_page(recommender):
    """Display the recommendations page"""
    st.markdown('<h2 class="sub-header">🎯 Get Fashion Recommendations</h2>', unsafe_allow_html=True)
    
    # Method selection with enhanced UI
    st.markdown('<div class="method-selector">', unsafe_allow_html=True)
    method = st.radio("Choose recommendation method:", 
                     ["🖼️ Upload Image (AI-Powered)", "📝 Text Search (Metadata-Based)"],
                     horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if method == "🖼️ Upload Image (AI-Powered)":
        st.markdown("### Upload an image to find visually similar fashion items")
        st.info("💡 Our GAT (Graph Attention Network) model analyzes visual features like colors, patterns, and styles.")
        
        # Upload area
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'],
                                        help="Upload a clear image of a fashion item for best results")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption="Uploaded Image", width=250)
                
                # Image info
                st.markdown("**Image Details:**")
                st.write(f"Size: {image.size}")
                st.write(f"Mode: {image.mode}")
            
            with col2:
                st.markdown("### Recommendation Settings")
                num_recommendations = st.slider("Number of recommendations:", 3, 15, 9)
                max_candidates = st.slider("Search scope (for faster results):", 100, 1000, 500, 
                                         help="Lower values = faster results, higher values = more comprehensive search")
                
                if st.button("🔍 Get AI Recommendations", type="primary", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing your image..."):
                        recommendations = recommender.recommend_by_image(image, num_recommendations, max_candidates)
                        display_recommendations(recommendations, recommender, "Optimized Image-based AI Analysis")
    
    else:  # Text Description
        st.markdown("### Describe the fashion item you're looking for")
        st.info("💡 Search uses metadata tags and categories. Use terms like colors, clothing types, styles.")
        
        # Text input with enhanced UI
        text_query = st.text_input("Enter your search query:", 
                                  placeholder="e.g., black shirt, red dress, denim, casual, formal",
                                  help="Search through clothing categories, colors, and style tags")
        
        # Quick suggestion buttons
        st.markdown("**💡 Quick Suggestions:**")
        suggestion_cols = st.columns(4)
        suggestions = [
            "shirt", "dress", "jacket", "top",
            "black", "white", "blue", "red"
        ]
        
        selected_suggestion = None
        for i, suggestion in enumerate(suggestions):
            with suggestion_cols[i % 4]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    selected_suggestion = suggestion
        
        if selected_suggestion:
            text_query = selected_suggestion
            st.rerun()
        
        # Search settings
        col1, col2 = st.columns(2)
        with col1:
            num_recommendations = st.slider("Number of recommendations:", 3, 15, 9, key="text_num_rec")
        with col2:
            st.markdown("**Search Method:** Direct metadata matching")
        
        if text_query:
            if st.button("🔎 Search Fashion Items", type="primary", use_container_width=True):
                with st.spinner("🔍 Searching through fashion database..."):
                    recommendations = recommender.recommend_by_text(text_query, num_recommendations)
                    display_recommendations(recommendations, recommender, f"Text Search: '{text_query}'")

def show_analytics_page(recommender):
    """Display analytics and insights"""
    st.markdown('<h2 class="sub-header">📊 Fashion Analytics</h2>', unsafe_allow_html=True)
    
    # Category distribution
    st.markdown("### 📈 Category Distribution")
    categories = {}
    for item in recommender.metadata:
        cat = item.get('category_name', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    # Create a DataFrame for better visualization
    df_categories = pd.DataFrame(list(categories.items()), columns=['Category', 'Count'])
    df_categories = df_categories.sort_values('Count', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.bar_chart(df_categories.set_index('Category'))
    
    with col2:
        st.dataframe(df_categories, use_container_width=True)
    
    # Tag analysis
    st.markdown("### 🏷️ Popular Tags")
    tag_counts = {}
    for item in recommender.metadata:
        if 'tag_info' in item:
            for tag in item['tag_info']:
                tag_cat = tag.get('tag_category', '')
                if tag_cat:
                    tag_counts[tag_cat] = tag_counts.get(tag_cat, 0) + 1
    
    # Top 20 tags
    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    df_tags = pd.DataFrame(top_tags, columns=['Tag', 'Count'])
    
    st.bar_chart(df_tags.set_index('Tag'))
    
    # Model performance info
    st.markdown("### 🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **GAT Model Features:**
        - Graph Attention Network architecture
        - Multi-head attention mechanism
        - Feature extraction capabilities
        - Similarity scoring
        """)
    
    with col2:
        st.markdown(f"""
        **Dataset Statistics:**
        - Total items: {len(recommender.metadata):,}
        - Available images: {sum(1 for item in recommender.metadata if (recommender.cloth_path / item['file_name']).exists()):,}
        - Categories: {len(set(item.get('category_name', 'Unknown') for item in recommender.metadata))}
        - Unique tags: {len(tag_counts)}
        """)

def display_template_grid(templates, recommender, columns=4):
    """Display templates in a grid layout with enhanced styling"""
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
                    st.markdown('<div class="template-card">', unsafe_allow_html=True)
                    
                    if img:
                        st.image(img, width=200)
                    else:
                        st.write("Image not found")
                    
                    st.markdown(f"**{item.get('category_name', 'Unknown')}**")
                    
                    # Show top tags with badges
                    if 'tag_info' in item:
                        tags = [tag.get('tag_category', '') for tag in item['tag_info'] 
                               if tag.get('tag_category')]
                        if tags:
                            tag_html = ""
                            for tag in tags[:3]:
                                tag_html += f'<span class="feature-badge">{tag}</span>'
                            st.markdown(tag_html, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

def display_recommendations(recommendations, recommender, search_type):
    """Display recommendation results with enhanced styling"""
    if not recommendations:
        st.error("🔍 No recommendations found. Try adjusting your search or uploading a different image.")
        return
    
    st.markdown(f"### 🎯 {search_type} Results")
    st.success(f"Found {len(recommendations)} recommendations!")
    
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
                        st.image(img, width=220)
                    else:
                        st.write("Image not available")
                    
                    # Display similarity score
                    score = item.get('similarity_score', 0)
                    st.markdown(f'<div class="similarity-score">Match: {score:.1%}</div>', 
                              unsafe_allow_html=True)
                    
                    # Display item details
                    st.markdown(f"**Category:** {item.get('category_name', 'Unknown')}")
                    
                    # Display tags with badges
                    if 'tag_info' in item:
                        tags_dict = {}
                        for tag in item['tag_info']:
                            tag_name = tag.get('tag_name', '')
                            tag_category = tag.get('tag_category', '')
                            if tag_name and tag_category:
                                tags_dict[tag_name] = tag_category
                        
                        if tags_dict:
                            for tag_name, tag_category in list(tags_dict.items())[:4]:
                                st.markdown(f"**{tag_name.title()}:** {tag_category}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
