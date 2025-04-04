import asyncio
import streamlit as st
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import io

# Ensure an event loop is available
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Model setup
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float16 for GPU, otherwise use float32 for CPU
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = "stabilityai/sdxl-turbo"
    return AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype).to(device)

pipeline = load_model()

# Interface setup
st.title("🖼️ EdgeAI Image Generator")

prompt = st.text_area("Enter your prompt:", "a happy black dog, cartoon style, highly detailed")

col1, col2 = st.columns(2)
width = col1.number_input("Width", min_value=256, max_value=1024, value=800, step=64)
height = col2.number_input("Height", min_value=256, max_value=1024, value=800, step=64)
steps = st.slider("Inference Steps", 1, 50, 10)

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        try:
            result = pipeline(prompt=prompt, width=width, height=height, num_inference_steps=steps)
            image = result.images[0]
            st.image(image, use_container_width=True)

            # Convert image to buffer
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            st.download_button("💾 Download Image", img_bytes, "generated.png", "image/png")
        except Exception as e:
            st.error(f"Error generating image: {e}")
