import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit.components.v1 as components

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("lora_finetuned_model")  # Ou use PEFT se necessário
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2_tokenizer")
    return model, tokenizer

model, tokenizer = load_model()

def generate_response(instruction, input_text="", max_tokens=50, temperature=0.7):
    prompt = f"Instruction: {instruction}\nOutput:" if not input_text else f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        do_sample=True  # necessário com temperature
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Output:")[-1].strip()

# Streamlit UI
st.title("Instruction-Based Text Generation")
st.write("This interface uses a fine-tuned language model to generate responses based on your instructions.")

# Input form
with st.form("generation_form"):
    instruction = st.text_area("Instruction*", 
                             placeholder="Enter your instruction here (e.g., 'How do I activate cruise control?')",
                             height=100)
    input_text = st.text_area("Additional Input (optional)", 
                             placeholder="Provide any additional context if needed",
                             height=100)
    
    col1, col2 = st.columns(2)
    with col1:
        max_tokens = st.slider("Max tokens", 20, 200, 50)
    with col2:
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
    
    submitted = st.form_submit_button("Generate Response")

# Generate and display response
if submitted:
    if not instruction:
        st.error("Please enter an instruction")
    else:
        with st.spinner("Generating response..."):
            response = generate_response(instruction, input_text, max_tokens, temperature)
        
        st.subheader("Generated Response")
        st.write(response)
        
        st.code(response, language="text")
        
        # Working copy button with JS
        components.html(f"""
        <button onclick="navigator.clipboard.writeText(`{response}`)">Copy to clipboard</button>
        """, height=40)
