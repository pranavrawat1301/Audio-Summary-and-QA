from audio_to_text import audio_file_to_text
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

TRANSCRIPTION_PATH = os.path.join(os.path.dirname(__file__), "transcription.txt")
FAISS_PATH = os.path.join(os.path.dirname(__file__), "faiss_local")
INDEX_NAME = "faiss_index"

def load_bart_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        framework="pt",  # Explicitly use PyTorch
        device=-1  # CPU, use 0 for GPU if available
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def save_documents_to_faiss(file_path):
    if os.path.exists(os.path.join(FAISS_PATH, f"{INDEX_NAME}.faiss")):
        print("FAISS index already exists. Skipping save.")
        return
    
    try:
        if os.path.exists(TRANSCRIPTION_PATH):
            with open(TRANSCRIPTION_PATH, "r", encoding='utf-8') as f:
                text_content = f.read()
        else:
            text_content = audio_file_to_text(file_path)
        
        document = Document(page_content=text_content, metadata={"filename": file_path})
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents([document])
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        os.makedirs(FAISS_PATH, exist_ok=True)
        db = FAISS.from_documents(split_docs, embeddings)
        db.save_local(folder_path=FAISS_PATH, index_name=INDEX_NAME)
        
    except Exception as e:
        print(f"Error in save_documents_to_faiss: {str(e)}")
        raise
