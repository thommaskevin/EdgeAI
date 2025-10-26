"""
Vision Transformer Backend for Image Classification
Handles model loading, preprocessing, and prediction logic
"""

import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image, ImageDraw
import numpy as np
import requests
import io

class VisionTransformerBackend:
    """
    Backend class for Vision Transformer operations
    Handles model management, image processing, and predictions
    """
    
    def __init__(self, model_name="google/vit-base-patch16-224"):
        """
        Initialize the Vision Transformer backend
        
        Steps:
        1. Set model name
        2. Initialize processor and model as None
        3. Load model when needed (lazy loading)
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.is_loaded = False
    
    def load_model(self):
        """
        Load the Vision Transformer model and processor
        
        Steps:
        1. Check if model is already loaded
        2. Load ViTImageProcessor for image preprocessing
        3. Load ViTForImageClassification for predictions
        4. Set loaded flag to True
        """
        if not self.is_loaded:
            try:
                # Step 1: Load the image processor (handles resizing, normalization)
                self.processor = ViTImageProcessor.from_pretrained(self.model_name)
                
                # Step 2: Load the Vision Transformer model
                self.model = ViTForImageClassification.from_pretrained(self.model_name)
                
                # Step 3: Set model to evaluation mode (no gradients)
                self.model.eval()
                
                self.is_loaded = True
                print("✅ Vision Transformer model loaded successfully!")
                
            except Exception as e:
                print(f"❌ Error loading model: {str(e)}")
                raise e
    
    def preprocess_image(self, image):
        """
        Preprocess image for Vision Transformer
        
        Steps:
        1. Convert image to RGB if needed
        2. Use ViT processor to:
           - Resize to 224x224
           - Normalize pixel values
           - Convert to tensor format
        """
        if not self.is_loaded:
            self.load_model()
        
        # Convert image to RGB if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Use ViT processor to preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs
    
    def predict(self, image):
        """
        Perform image classification using Vision Transformer
        
        Steps:
        1. Preprocess the input image
        2. Run inference (no gradients for efficiency)
        3. Apply softmax to get probabilities
        4. Extract top 5 predictions
        5. Format results with labels and probabilities
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Step 1: Preprocess image
            inputs = self.preprocess_image(image)
            
            # Step 2: Run inference (no gradient calculation for efficiency)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Step 3: Convert logits to probabilities using softmax
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Step 4: Get top 5 predictions
            top_probs, top_indices = torch.topk(probabilities, 5)
            
            # Step 5: Convert tensors to Python lists
            top_probs = top_probs[0].tolist()
            top_indices = top_indices[0].tolist()
            
            # Step 6: Format predictions with human-readable labels
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                label = self.model.config.id2label[idx]
                predictions.append({
                    'label': label,
                    'probability': prob,
                    'percentage': prob * 100,
                    'confidence': f"{prob * 100:.2f}%"
                })
            
            return predictions
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            return None
    
    def visualize_patches(self, image, patch_size=16):
        """
        Visualize how ViT divides image into patches
        
        Steps:
        1. Resize image to 224x224 (ViT input size)
        2. Create copy for drawing
        3. Draw grid lines for each 16x16 patch
        4. Return visualized image
        """
        # Resize to model input size
        processed_image = image.resize((224, 224))
        
        # Create copy for drawing patches
        patch_viz = processed_image.copy()
        draw = ImageDraw.Draw(patch_viz)
        
        # Draw patch grid lines
        for i in range(0, 224, patch_size):
            for j in range(0, 224, patch_size):
                # Draw rectangle for each patch
                draw.rectangle([j, i, j+patch_size, i+patch_size], 
                             outline='red', width=1)
        
        return patch_viz
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model specifications
        """
        if not self.is_loaded:
            self.load_model()
        
        return {
            'name': self.model_name,
            'input_size': '224x224',
            'patch_size': 16,
            'num_patches': (224 // 16) * (224 // 16),
            'num_classes': self.model.config.num_labels,
            'model_type': 'Vision Transformer (ViT)'
        }
    
    def load_image_from_url(self, url):
        """
        Load image from URL for processing
        
        Steps:
        1. Send HTTP request to URL
        2. Read image data
        3. Convert to PIL Image
        4. Ensure RGB format
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            print(f"❌ Error loading image from URL: {str(e)}")
            return None
    
    def load_image_from_file(self, file_path):
        """
        Load image from local file
        
        Steps:
        1. Open file using PIL
        2. Convert to RGB format
        3. Return image object
        """
        try:
            image = Image.open(file_path).convert('RGB')
            return image
        except Exception as e:
            print(f"❌ Error loading image from file: {str(e)}")
            return None

# Utility function to create prediction visualization
def create_prediction_chart(predictions):
    """
    Create matplotlib visualization for predictions
    
    Steps:
    1. Extract labels and probabilities
    2. Create horizontal bar chart
    3. Apply color gradient
    4. Add value labels
    """
    import matplotlib.pyplot as plt
    
    labels = [pred['label'].replace('_', ' ').title() for pred in predictions]
    probabilities = [pred['percentage'] for pred in predictions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    
    # Create color gradient
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
    
    # Create horizontal bars
    bars = ax.barh(y_pos, probabilities, color=colors)
    
    # Configure chart appearance
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()  # Highest probability at top
    ax.set_xlabel('Probability (%)', fontsize=12)
    ax.set_title('Top 5 Predictions - Vision Transformer', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
               f'{width:.2f}%', ha='left', va='center', 
               fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    # Test the backend
    backend = VisionTransformerBackend()
    
    # Test with a sample image
    test_url = "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400"
    test_image = backend.load_image_from_url(test_url)
    
    if test_image:
        predictions = backend.predict(test_image)
        print("Predictions:", predictions)