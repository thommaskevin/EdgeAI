"""
Streamlit Frontend for Vision Transformer Image Classification
Handles user interface, interactions, and visualization
"""

import streamlit as st
import time
import os
from vit_backend import VisionTransformerBackend, create_prediction_chart
from PIL import Image
import io

# Configure page settings
st.set_page_config(
    page_title="Vision Transformer Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize backend (cached to avoid reloading)
@st.cache_resource
def initialize_backend():
    """
    Initialize the Vision Transformer backend with caching
    
    Steps:
    1. Create backend instance
    2. Model loads automatically on first use
    3. Cached to avoid reloading on every interaction
    """
    return VisionTransformerBackend()

def setup_sidebar(backend):
    """
    Configure the sidebar with model information and controls
    
    Steps:
    1. Display model information
    2. Show technical details
    3. Provide user guidance
    """
    st.sidebar.title("⚙️ Configuration")
    
    # Display model information
    st.sidebar.markdown("### Model Information")
    model_info = backend.get_model_info()
    st.sidebar.info(f"""
    **Model:** {model_info['name'].split('/')[-1]}
    **Input Size:** {model_info['input_size']}
    **Patches:** {model_info['patch_size']}x{model_info['patch_size']}
    **Parameters:** 86M
    """)
    
    # Technical details expander
    with st.sidebar.expander("📊 About Vision Transformer"):
        st.markdown("""
        **Architecture Steps:**
        1. **Image → Patches**: Divide into 16x16 patches
        2. **Linear Embedding**: Flatten and project patches
        3. **Position Encoding**: Add spatial information
        4. **Transformer Encoder**: Process with self-attention
        5. **Classification Head**: Generate final predictions
        
        **Key Advantages:**
        - 🎯 Global attention mechanism
        - 📈 Excellent scalability
        - 🏆 State-of-the-art performance
        """)
    
    # Usage tips
    with st.sidebar.expander("💡 Usage Tips"):
        st.markdown("""
        **Best Results With:**
        - Clear, well-lit images
        - Common objects and animals
        - Single dominant subject
        - Good contrast and focus
        
        **Try These Categories:**
        - Animals 🐕🐈🦜
        - Vehicles 🚗✈️🚂
        - Food 🍕🍔🍎
        - Landmarks 🗼🏛️🌉
        """)

def handle_image_input():
    """
    Handle different image input methods from user
    
    Steps:
    1. Provide multiple input options (upload, URL, samples)
    2. Validate and load images
    3. Return loaded image or None
    """
    st.header("📤 Input Image")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Use Sample Image", "Enter Image URL"],
        horizontal=True
    )
    
    image = None
    
    if input_method == "Upload Image":
        # File uploader widget
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Select an image file from your device"
        )
        if uploaded_file is not None:
            # Load and validate uploaded image
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.success("✅ Image uploaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
    
    elif input_method == "Use Sample Image":
        # Pre-defined sample images
        sample_options = {
            "Golden Retriever": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400",
            "Siamese Cat": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
            "Sports Car": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=400",
            "Pizza": "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=400",
            "Eiffel Tower": "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?w=400"
        }
        
        selected_sample = st.selectbox("Choose sample image:", list(sample_options.keys()))
        
        if st.button("🖼️ Load Sample Image", use_container_width=True):
            # Load sample image with progress indication
            with st.spinner("Loading sample image..."):
                image = backend.load_image_from_url(sample_options[selected_sample])
                if image:
                    st.success("✅ Sample image loaded!")
                else:
                    st.error("❌ Failed to load sample image")
    
    elif input_method == "Enter Image URL":
        # URL input method
        url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")
        
        if st.button("🌐 Load from URL", use_container_width=True):
            if url:
                with st.spinner("Loading image from URL..."):
                    image = backend.load_image_from_url(url)
                    if image:
                        st.success("✅ Image loaded from URL!")
                    else:
                        st.error("❌ Failed to load image from URL")
            else:
                st.warning("⚠️ Please enter a valid URL")
    
    return image

def display_image_info(image):
    """
    Display image metadata and properties
    
    Steps:
    1. Show original image
    2. Display technical information
    3. Provide image statistics
    """
    if image is not None:
        # Display the input image
        st.image(image, caption="Original Input Image", use_container_width=True)
        
        # Image metadata section
        st.subheader("📊 Image Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Width", f"{image.width}px")
        with col2:
            st.metric("Height", f"{image.height}px")
        with col3:
            st.metric("Mode", image.mode)

def create_progress_animation():
    """
    Create visual progress indicator during classification
    
    Steps:
    1. Initialize progress bar
    2. Update status messages
    3. Simulate processing steps
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate processing steps with progress updates
    for i in range(101):
        progress_bar.progress(i)
        if i < 25:
            status_text.text("🔄 Preprocessing image...")
        elif i < 50:
            status_text.text("🔍 Extracting patches...")
        elif i < 75:
            status_text.text("🧠 Processing through transformer...")
        else:
            status_text.text("📊 Generating predictions...")
        time.sleep(0.02)
    
    status_text.text("✅ Classification complete!")
    return progress_bar, status_text

def display_classification_process(backend, image):
    """
    Display the step-by-step classification process
    
    Steps:
    1. Show image preprocessing
    2. Visualize patch extraction
    3. Display final results
    """
    st.header("🔍 Classification Results")
    st.subheader("🔄 Classification Process")
    
    # Step 1: Image Preprocessing
    with st.expander("Step 1: Image Preprocessing", expanded=True):
        st.info("Resizing image to 224x224 and normalizing pixel values...")
        processed_image = image.resize((224, 224))
        
        # Show before/after comparison
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption=f"Original ({image.width}x{image.height})", use_container_width=True)
        with col2:
            st.image(processed_image, caption="Preprocessed (224x224)", use_container_width=True)
    
    # Step 2: Patch Extraction Visualization
    with st.expander("Step 2: Patch Extraction", expanded=True):
        st.info("Dividing image into 16x16 patches for transformer processing...")
        patch_viz = backend.visualize_patches(image)
        st.image(patch_viz, caption="16x16 Patch Grid Visualization", use_container_width=True)
        st.caption(f"Total patches: {(224//16) * (224//16)} = 196 patches")
    
    # Step 3: Classification Execution
    if st.button("🚀 Classify Image", type="primary", use_container_width=True):
        # Show progress animation
        progress_bar, status_text = create_progress_animation()
        
        # Perform actual classification
        predictions = backend.predict(image)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if predictions:
            display_results(predictions)

def display_results(predictions):
    """
    Display classification results in an organized format
    
    Steps:
    1. Show top prediction prominently
    2. Display prediction distribution chart
    3. List detailed predictions with confidence
    """
    with st.expander("Step 3: Classification Results", expanded=True):
        st.success("✅ Classification completed!")
        
        # Highlight top prediction
        top_pred = predictions[0]
        st.metric(
            "🎯 Top Prediction", 
            top_pred['label'].replace('_', ' ').title(),
            f"{top_pred['percentage']:.2f}%"
        )
        
        # Prediction distribution chart
        st.subheader("📈 Prediction Distribution")
        fig = create_prediction_chart(predictions)
        st.pyplot(fig)
        
        # Detailed predictions list
        st.subheader("📋 Detailed Predictions")
        for i, pred in enumerate(predictions):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 2])
                with col1:
                    st.write(f"**{i+1}. {pred['label'].replace('_', ' ').title()}**")
                with col2:
                    st.write(f"**{pred['percentage']:.2f}%**")
                with col3:
                    st.progress(float(pred['probability']), 
                               text=f"Confidence: {pred['percentage']:.1f}%")
            if i < len(predictions) - 1:
                st.divider()

def display_welcome_message():
    """
    Display welcome message and instructions when no image is loaded
    """
    st.info("👆 Please provide an image using one of the input methods on the left.")
    st.markdown("""
    **💡 Try these sample categories:**
    - 🐕 Animals (dogs, cats, birds)
    - 🚗 Vehicles (cars, airplanes, boats)
    - 🍕 Food (pizza, hamburgers, fruits)
    - 🏛️ Landmarks (buildings, monuments)
    - 🏠 Everyday objects
    """)

def main():
    """
    Main application function
    
    Steps:
    1. Initialize backend
    2. Setup sidebar
    3. Handle image input
    4. Process and display results
    """
    # Application title and description
    st.title("🔍 Vision Transformer Image Classification")
    st.markdown("""
    Experience the power of **Transformer architecture** applied to computer vision! 
    This application uses Google's Vision Transformer (ViT) to classify images with state-of-the-art accuracy.
    """)
    
    # Initialize backend (cached)
    global backend
    backend = initialize_backend()
    
    # Setup sidebar configuration
    setup_sidebar(backend)
    
    # Main content layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Handle image input from user
        image = handle_image_input()
        
        # Display image information if loaded
        if image is not None:
            display_image_info(image)
    
    with col2:
        if image is not None:
            # Show classification process and results
            display_classification_process(backend, image)
        else:
            # Show welcome message
            display_welcome_message()
    
    # Application footer
    st.markdown("---")
    st.markdown(
        "**Technical Details:** Powered by Google's Vision Transformer (ViT-Base) from Hugging Face. "
        "Model pre-trained on ImageNet-21k and fine-tuned on ImageNet 2012. "
        "Architecture: 12 layers, 768 hidden dimensions, 12 attention heads."
    )

# Global backend instance
backend = None

if __name__ == "__main__":
    # Set environment variable to suppress warnings
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    
    # Run the main application
    main()