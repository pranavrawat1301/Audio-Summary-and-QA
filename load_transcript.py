from audio_to_text import audio_file_to_text

import os
import torch
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import assemblyai as aai
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


TRANSCRIPTION_PATH = r"D:\Projects(internship)\LLM based Projects\Audio Summarizer\transcription.txt"
FAISS_PATH = r"D:\Projects(internship)\LLM based Projects\Audio Summarizer\faiss_local"
INDEX_NAME = "faiss_index"


def load_bart_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def save_documents_to_faiss(file_path):

    if os.path.exists(os.path.join(FAISS_PATH, f"{INDEX_NAME}.faiss")):
        print("FAISS index already exists. Skipping save.")
        return
    
    
    if os.path.exists(TRANSCRIPTION_PATH):
        with open(TRANSCRIPTION_PATH, "r") as f:
            text_content = f.read()
    else:
        text_content = audio_file_to_text(file_path)
    
    document = Document(page_content=text_content, metadata={"filename": file_path})
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    split_docs = text_splitter.split_documents([document])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(folder_path=FAISS_PATH, index_name=INDEX_NAME)