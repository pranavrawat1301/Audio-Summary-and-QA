from audio_to_text import audio_file_to_text
from transformers import pipeline
import streamlit as st
import torch

@st.cache_resource
def load_summarizer():
    """Cache the summarization model"""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

@st.cache_data
def response_generator(file_path):
    """Cache summary generation for same audio file"""
    searchDocs = audio_file_to_text(file_path)
    
    # Optimize chunk size and processing
    chunk_size = 512  # Smaller chunks for faster processing
    chunks = [searchDocs[i:i+chunk_size] for i in range(0, len(searchDocs), chunk_size)]
    
    summarizer = load_summarizer()
    summaries = []
    
    # Process chunks in batches for better performance
    batch_size = 4
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_summaries = summarizer(
            batch,
            max_length=100,
            min_length=30,
            do_sample=False,
            batch_size=batch_size
        )
        summaries.extend([s['summary_text'] for s in batch_summaries])
    
    return " ".join(summaries)



