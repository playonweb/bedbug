import streamlit as st
import torch
from transformers import pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="TinyEdge Chat",
    page_icon="🤖",
    layout="wide",
)

st.title("TinyEdge Chat 🤖")
st.markdown("A lightweight, edge-capable Chatbot powered by TinyLlama.")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_model():
    """Loads the TinyLlama model pipeline. Cached to prevent reloading on every interaction."""
    # Using pipeline for simplicity. We use text-generation pipeline.
    # TinyLlama is small enough to run on CPU with bfloat16 or just float32.
    pipe = pipeline(
        "text-generation", 
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    return pipe

try:
    with st.spinner("Loading AI Model (This might take a moment on the first run)..."):
        pipe = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Format messages for TinyLlama
    # Using the tokenizer's apply_chat_template to format the conversation history correctly
    try:
        prompt_formatted = pipe.tokenizer.apply_chat_template(
            st.session_state.messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception as e:
        st.error(f"Error formatting chat template: {e}")
        st.stop()
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            try:
                # Generate response
                outputs = pipe(
                    prompt_formatted,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                )
                
                # Extract just the newly generated text
                full_text = outputs[0]["generated_text"]
                response = full_text[len(prompt_formatted):]
                
                message_placeholder.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating response: {e}")
