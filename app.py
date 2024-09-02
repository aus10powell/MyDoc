import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

@st.cache_resource
def load_model():
    model_name = "aaditya/OpenBioLLM-Llama3-8B-GGUF"
    model_file = "openbiollm-llama3-8b.Q5_K_M.gguf"
    model_path = hf_hub_download(model_name, filename=model_file)
    return Llama(model_path=model_path, n_gpu_layers=-1)

llm = load_model()

st.title("OpenBioLLM Chat Interface")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if question := st.chat_input("What is your medical question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    # Construct the prompt
    prompt = f"""You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs with Open Life Science AI. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience. Medical Question: {question} Medical Answer:"""

    # Generate response
    response = llm(prompt, max_tokens=4000)['choices'][0]['text']
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})