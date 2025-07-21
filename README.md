

## Introduction

The rapid growth of virtual fashion technologies has revolutionized the landscape of e-commerce, digital wardrobe applications, and personalized styling assistants. With the increasing demand for immersive, intelligent clothing experiences, a unified system that supports both visual understanding and interaction is critical. This research introduces a comprehensive, multi-modal framework that addresses this need by integrating three key components: (1) a Graph Attention Network (GAT)-based fashion recommendation engine, (2) a Dense Correspondence Inpainting Virtual Try-On (DCI-VTON) pipeline, and (3) a novel GAN-based Try-Off system for clothing extraction.

**1. GAT-Based Fashion Recommendation**
At the core of our recommendation module lies a multi-layer Graph Attention Network that leverages structured visual-language features extracted from clothing images using Vision-Language Models (VLMs) such as InstructBLIP and MiniLM. Each item in the graph is embedded as a node with rich semantic attributes—e.g., collar type, sleeve style, pattern complexity—enabling the system to learn fine-grained inter-item relationships. The model is trained using contrastive and triplet loss functions, optimizing the embedding space for similarity-based retrieval. This allows the system to suggest contextually appropriate and stylistically coherent garments, even across diverse fashion domains.

**2. DCI-VTON-Based Virtual Try-On**
Our virtual try-on component builds upon the Dense Correspondence Inpainting (DCI) paradigm to generate high-fidelity visualizations of users wearing target garments. By combining appearance flow-based warping with spatial inpainting modules and leveraging DensePose or OpenPose keypoints for body structure alignment, DCI-VTON achieves robust garment transfer across complex poses and occlusions. This module ensures that clothing retains its original texture, structure, and photorealistic appearance while conforming naturally to the user’s silhouette. The system can simulate multiple try-on scenarios, offering a dynamic and interactive fitting room experience.

**3. GAN-Based Virtual Try-Off**
Complementing the try-on process is our novel GAN-based try-off pipeline, which performs the inverse task—removing the worn clothing from person images and recovering its canonical (flat) appearance. This module uses a dedicated generator-discriminator architecture trained on isolated and worn garment pairs to learn the mapping from draped to flat representations. This enables realistic garment extraction that preserves texture details and spatial coherence, making it suitable for catalog generation, garment reusability, and further try-on tasks. When integrated with the VTON pipeline, this component forms a bidirectional dressing system that greatly enhances user flexibility.

By unifying these modules into a single system, our framework bridges the gap between intelligent garment retrieval and high-fidelity image synthesis. It opens new possibilities in digital fashion retail, styling assistance, content creation, and augmented reality experiences—marking a significant advancement in fashion-focused AI research.

**Install dependencies**
pip install -r requirements.txt

**Usage**
cd recommendation
python train_gat.py    
python recommend.py    

