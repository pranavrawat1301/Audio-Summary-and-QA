from audio_to_text import audio_file_to_text
import os
import torch
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import streamlit as st

TRANSCRIPTION_PATH = "transcription.txt"
FAISS_PATH = "faiss_local"
INDEX_NAME = "faiss_index"

@st.cache_resource
def load_bart_model():
    """Cache the BART model loading to avoid reloading for each query"""
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

@st.cache_data
def save_documents_to_faiss(_file_path, text_content):
    """Cache FAISS index creation based on file content"""
    document = Document(page_content=text_content, metadata={"filename": _file_path})
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # Reduced chunk size for faster processing
        chunk_overlap=50  # Reduced overlap
    )
    split_docs = text_splitter.split_documents([document])
    
    # Cache embeddings model
    embeddings = get_embeddings()
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(folder_path=FAISS_PATH, index_name=INDEX_NAME)
    return db

@st.cache_resource
def get_embeddings():
    """Cache the embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
