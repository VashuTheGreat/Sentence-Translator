import streamlit as st
from modelOutput import load_model_files, translate  # renamed loader

# Page config
st.set_page_config(page_title="🌐 Sentence Translator", page_icon="🌍", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .stSelectbox label {
        font-weight: bold;
        color: #2E4053;
    }
    .result-box {
        background-color: #F2F3F4;
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        color: #1A5276;
        font-weight: 500;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title & description
st.markdown("<div class='title'>🌐 Sentence Translator</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Translate your sentences instantly between languages</div>", unsafe_allow_html=True)

# Language selection
options = ['French', 'Hindi']
selected_element = st.selectbox('🌍 Choose the target language:', options)

# Load model only when language changes
if 'current_lang' not in st.session_state or st.session_state.current_lang != selected_element:
    load_model_files(selected_element)
    st.session_state.current_lang = selected_element

# Input text
title = st.text_input("✏️ Enter a sentence to translate:")

# Translation output
if title:
    pred = translate(title)
    st.markdown(f"<div class='result-box'>✅ <b>Translated:</b> {pred}</div>", unsafe_allow_html=True)
