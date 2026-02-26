import sys
import os
import streamlit as st
import asyncio
import pandas as pd
import nltk
try:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
except Exception as e:
    print(e)
    

# Add project root to path
sys.path.append(os.getcwd())

from src.pipelines.Prediction_Pipeline import PredictionPipeline

st.set_page_config(
    page_title="Sentence Translator",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 English → Hindi Translator")
st.markdown("Translate English sentences into Hindi using your trained model.")

@st.cache_resource  
def get_prediction_pipeline():
    return PredictionPipeline()

prediction_pipeline = get_prediction_pipeline()

# Example Sentences
examples = [
    "Give your application an accessibility workout",
    "Accerciser Accessibility Explorer",
    "The default plugin layout for the bottom panel",
    "The default plugin layout for the top panel",
    "A list of plugins that are disabled by default",
    "Highlight duration",
    "The duration of the highlight box when selecting accessible nodes",
    "Highlight border color",
    "The color and opacity of the highlight border.",
    "Highlight fill color",
    "The color and opacity of the highlight fill.",
    "API Browser",
    "Browse the various methods of the current accessible",
    "Hide private attributes",
    "Method",
    "Property",
    "Value"
]

# Convert to DataFrame for table display
df = pd.DataFrame({"Example Sentences": examples})

st.subheader("📌 Pick an Example Sentence")
st.markdown("Click any row below to auto-fill the text box.")



# Alternative: Selectbox for auto-fill
selected_example = st.selectbox(
    "Or choose from dropdown:",
    ["-- Select Example --"] + examples
)

# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# If user selects example
if selected_example != "-- Select Example --":
    st.session_state.input_text = selected_example

# Text Area
input_text = st.text_area(
    "✍️ Enter English Sentence",
    value=st.session_state.input_text,
    height=200
)

# Translate Button
if st.button("🚀 Translate"):
    if input_text.strip() == "":
        st.warning("Please enter a sentence first.")
    else:
        with st.spinner("Translating..."):
            result = asyncio.run(
                prediction_pipeline.initiate_prediction_pipeline(input_text)
            )
        st.success("Translation Complete!")
        st.write("### 📝 Translated Sentence:")
        st.write(result)