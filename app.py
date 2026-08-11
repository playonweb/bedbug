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
st.markdown("A lightweight, edge-capable Chatbot.")

# --- Sidebar ---
st.sidebar.title("Model Settings")
st.sidebar.markdown("Select a tiny LLM that fits within edge device storage constraints (< 1.4GB).")

MODELS = {
    "SmolLM 135M (Ultra Tiny, ~270MB)": "HuggingFaceTB/SmolLM-135M-Instruct",
    "SmolLM 360M (Very Tiny, ~720MB)": "HuggingFaceTB/SmolLM-360M-Instruct",
    "Qwen2.5 0.5B (Tiny, ~1GB)": "Qwen/Qwen2.5-0.5B-Instruct"
}

selected_model_display = st.sidebar.selectbox("Choose AI Model", list(MODELS.keys()))
selected_model_id = MODELS[selected_model_display]

# Warn about disk space if changing models frequently
st.sidebar.info("Note: Downloading multiple models will consume more disk space. The first message on a new model will trigger a download.")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_model(model_id):
    """Loads the model pipeline. Cached per model_id to prevent reloading on every interaction."""
    # Using pipeline for simplicity. We use text-generation pipeline.
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    return pipe

try:
    with st.spinner(f"Loading {selected_model_display} (This might take a moment on the first run)..."):
        pipe = load_model(selected_model_id)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Chat Interface ---

# Initialize chat history (Clear history if model changes)
if "current_model" not in st.session_state or st.session_state.current_model != selected_model_id:
    st.session_state.messages = []
    st.session_state.current_model = selected_model_id

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
    
    # Format messages for the chosen model using its specific chat template
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
