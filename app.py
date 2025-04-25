import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import nltk
nltk.download('punkt')

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("model")
    model = AutoModelForSeq2SeqLM.from_pretrained("model")
    return tokenizer, model

tokenizer, model = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Constants
MAX_INPUT_LENGTH = 384
MAX_TARGET_LENGTH = 48

def clean_text(text):
    """Clean text by removing HTML tags and extra whitespace"""
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_summary(text):
    """Generate summary for the given text"""
    # Clean the text
    cleaned_text = clean_text(text)

    # Tokenize
    inputs = tokenizer(
        cleaned_text, 
        max_length=MAX_INPUT_LENGTH, 
        truncation=True, 
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_length=MAX_TARGET_LENGTH,
            min_length=10,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit UI
st.title("Amazon Review Summarizer")
st.write("This app generates concise summaries of Amazon product reviews using a fine-tuned BART model.")

review_text = st.text_area(
    "Enter your product review:", 
    "I recently purchased this wireless speaker, and I'm impressed with the sound quality! The bass is deep, and the treble is clear. It pairs easily with my phone, and the Bluetooth range is excellent...",
    height=200
)

if st.button("Generate Summary"):
    if review_text.strip():
        with st.spinner("Generating summary..."):
            summary = generate_summary(review_text)
            st.subheader("Generated Summary:")
            st.success(summary)
    else:
        st.warning("Please enter a review to summarize.")