import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="GenAI Suite", layout="wide")

API_URL = "http://localhost:8000"

st.title("🤖 Unified Generative AI Suite")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["GPT-2", "Markov", "Stable Diffusion", "Pix2Pix", "Style Transfer"])

# Tab 1: GPT-2
with tab1:
    st.header("GPT-2 Text Generation")
    prompt = st.text_area("Prompt", "Once upon a time...")
    col1, col2 = st.columns(2)
    with col1:
        max_len = st.slider("Max Length", 10, 200, 60)
        temp = st.slider("Temperature", 0.1, 2.0, 0.9)
    with col2:
        top_k = st.slider("Top-K", 1, 100, 50)
        top_p = st.slider("Top-P", 0.1, 1.0, 0.95)
    
    if st.button("Generate Text", key="gpt2"):
        with st.spinner("Generating..."):
            response = requests.post(f"{API_URL}/generate_text_gpt2", params={
                "prompt": prompt, "max_length": max_len, "temperature": temp,
                "top_k": top_k, "top_p": top_p
            })
            st.text_area("Output", response.json()["result"], height=300)

# Tab 2: Markov
with tab2:
    st.header("Markov Chain Generator")
    corpus = st.text_area("Training Corpus", height=150)
    if st.button("Train Model"):
        requests.post(f"{API_URL}/train_markov", json={"corpus": corpus})
        st.success("Model trained!")
    
    length = st.slider("Output Length", 10, 200, 50)
    if st.button("Generate", key="markov"):
        response = requests.post(f"{API_URL}/generate_text_markov", params={"length": length})
        st.text_area("Output", response.json()["result"], height=200)

# Tab 3: Stable Diffusion
# Tab 3: Stable Diffusion
with tab3:
    st.header("Stable Diffusion Image Generation")
    sd_prompt = st.text_area("Prompt", "A beautiful sunset...")
    col1, col2 = st.columns(2)
    with col1:
        steps = st.slider("Inference Steps", 10, 100, 30)
    with col2:
        scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
    
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            response = requests.post(f"{API_URL}/generate_image", params={
                "prompt": sd_prompt, "steps": steps, "scale": scale
            })
            img = Image.open(io.BytesIO(response.content))
            st.image(img, caption="Generated Image")  # FIXED

# Tab 4: Pix2Pix
with tab4:
    st.header("Image-to-Image Translation")
    uploaded = st.file_uploader("Upload Input Image", type=['png', 'jpg', 'jpeg'])
    if uploaded and st.button("Translate"):
        with st.spinner("Translating..."):
            response = requests.post(f"{API_URL}/translate_image", 
                files={"file": uploaded})
            img = Image.open(io.BytesIO(response.content))
            st.image(img, caption="Translated Image")  # FIXED

# Tab 5: Style Transfer
with tab5:
    st.header("Neural Style Transfer")
    col1, col2 = st.columns(2)
    with col1:
        content = st.file_uploader("Content Image", type=['png', 'jpg'])
    with col2:
        style = st.file_uploader("Style Image", type=['png', 'jpg'])
    
    c_weight = st.slider("Content Weight", 1e3, 1e6, 1e5, format="%.0f")
    s_weight = st.slider("Style Weight", 1e7, 1e12, 1e10, format="%.0f")
    steps_st = st.slider("Optimization Steps", 100, 1000, 300)
    
    if content and style and st.button("Stylize"):
        with st.spinner("Applying style transfer... (1-3 minutes)"):
            try:
                content.seek(0)
                style.seek(0)
                response = requests.post(f"{API_URL}/style_transfer", 
                    files={"content_file": content, "style_file": style},
                    params={"content_weight": c_weight, "style_weight": s_weight, "steps": steps_st},
                    timeout=600
                )
                
                if response.status_code == 200:
                    img = Image.open(io.BytesIO(response.content))
                    st.image(img, caption="Stylized Result")  # FIXED
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
