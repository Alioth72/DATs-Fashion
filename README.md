# Fashion Recommender System

This project contains a frontend (React + Tailwind CSS) and a backend (Flask) for a fashion recommender system using local images and a GAT model.

## Project Structure

```
# Fashion Recommender System 👗🤖

An AI-powered fashion recommendation system using Graph Attention Networks (GAT) and advanced image processing techniques.

## Features

- **🧠 GAT Neural Network**: Advanced Graph Attention Network for precise image similarity matching
- **🔍 Smart Search**: Intelligent text-based search using metadata and semantic analysis
- **🎨 Curated Gallery**: Hand-picked templates showcasing diverse fashion styles
- **📊 Analytics Dashboard**: Insights into fashion trends and dataset statistics

## Dataset

- **11,467 fashion images** with comprehensive metadata
- **Diverse categories**: TOPS, BOTTOMS, DRESSES, OUTERWEAR, etc.
- **Rich metadata**: Colors, materials, styles, fits, and more
- **Template gallery**: 30 carefully selected diverse fashion items

## Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **Backend**: Python with PyTorch
- **AI Model**: Graph Attention Network (GAT)
- **Image Processing**: OpenCV, PIL
- **Text Processing**: TF-IDF, scikit-learn
- **Data**: JSON metadata with 11,467+ fashion items

## Installation

### Option 1: Quick Start (Windows)
1. Double-click `install.bat` to install dependencies
2. Double-click `run_app.bat` to start the application

### Option 2: Manual Installation
```bash
# Clone or download the project
cd clothes_tryon_dataset

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run enhanced_fashion_app.py
```

### Option 3: Using Virtual Environment
```bash
# Create virtual environment
python -m venv fashion_env

# Activate virtual environment
# On Windows:
fashion_env\Scripts\activate
# On macOS/Linux:
source fashion_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run enhanced_fashion_app.py
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster GAT model inference)
- 4GB+ RAM recommended
- Modern web browser

## File Structure

```
clothes_tryon_dataset/
├── enhanced_fashion_app.py      # Main Streamlit application
├── gat_model_integration.py     # GAT model implementation
├── requirements.txt             # Python dependencies
├── install.bat                  # Windows installation script
├── run_app.bat                 # Windows run script
├── gat_model.pth               # Pre-trained GAT model weights
├── vitonhd_train_tagged.json   # Fashion metadata (11,467 items)
├── test/cloth/                 # Fashion images directory
└── README.md                   # This file
```

## Usage

### 1. Browse Templates
- Explore 30 curated fashion templates
- Filter by category
- Sort by different criteria

### 2. Image-Based Recommendations
- Upload a fashion image
- Get AI-powered recommendations using GAT model
- View similarity scores and detailed item information

### 3. Text-Based Recommendations
- Describe the fashion item you're looking for
- Use natural language queries
- Get semantically similar fashion items

### 4. Analytics Dashboard
- View dataset statistics
- Analyze category distributions
- Explore popular fashion tags

## Model Information

### GAT (Graph Attention Network)
- **Architecture**: Multi-head attention mechanism
- **Input**: Fashion image features + metadata
- **Output**: Similarity scores and recommendations
- **Features**: Color analysis, texture recognition, style matching

### Fallback Model
- **Color histogram analysis**
- **Texture feature extraction**
- **Cosine similarity matching**
- **Used when GAT model is not available**

## Troubleshooting

### Common Issues

1. **GAT Model not loading**
   - Ensure `gat_model.pth` exists in the project directory
   - Check CUDA compatibility if using GPU
   - Application will use fallback model automatically

2. **Images not displaying**
   - Verify image files exist in `test/cloth/` directory
   - Check file permissions
   - Ensure image formats are supported (JPG, PNG)

3. **Slow performance**
   - Close other applications to free up memory
   - Consider using CPU-only mode if GPU is causing issues
   - Reduce number of recommendations for faster results

## Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Review the code documentation
3. Create an issue with detailed description

---

**Happy Fashion Recommending!** 👗✨
  frontend/   # React + Tailwind CSS app
  backend/    # Flask backend with GAT model
  train/cloth # Clothing images used for recommendations
```

---

## Frontend Setup (React + Tailwind)

1. Navigate to the project directory:
   ```sh
   cd clothes_tryon_dataset
   ```
2. Create the React app:
   ```sh
   npx create-react-app frontend
   cd frontend
   ```
3. Install Tailwind CSS:
   ```sh
   npm install -D tailwindcss postcss autoprefixer
   npx tailwindcss init -p
   ```
4. Configure Tailwind in `tailwind.config.js` and add Tailwind to `src/index.css`:
   - Add the following to `tailwind.config.js`:
     ```js
     content: ["./src/**/*.{js,jsx,ts,tsx}"]
     ```
   - In `src/index.css`, add:
     ```css
     @tailwind base;
     @tailwind components;
     @tailwind utilities;
     ```
5. Start the frontend:
   ```sh
   npm start
   ```

---

## Backend Setup (Flask)

1. Navigate to the backend directory:
   ```sh
   cd clothes_tryon_dataset
   mkdir backend
   cd backend
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install Flask and other dependencies:
   ```sh
   pip install flask flask-cors pillow
   ```
4. Place `app.py` in the `backend/` directory (provided in this repo).
5. Start the backend:
   ```sh
   python app.py
   ```

---

## Connecting Frontend and Backend

- The frontend will send requests to the backend at `http://localhost:5000` (default Flask port).
- Make sure both servers are running for full functionality.

---

## Notes
- Trending and recommended images are loaded from `train/cloth/`.
- The backend is ready for you to plug in your GAT model for real recommendations.
- For development, CORS is enabled in Flask for local testing. 