from load_transcript import load_bart_model

import os
import tensorflow as tf
import warnings

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only log errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='langchain')

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



import os

# Use a relative path
TRANSCRIPTION_PATH = os.path.join(os.getcwd(), "transcription.txt")
FAISS_PATH = os.path.join(os.getcwd(), "faiss_local")
INDEX_NAME = "faiss_index"


def similarity_search(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    db = FAISS.load_local(folder_path=FAISS_PATH, embeddings=embeddings, index_name=INDEX_NAME, allow_dangerous_deserialization=True)
    
    search_docs = db.similarity_search(user_question)
    return search_docs


def get_qa_chain():
    prompt_template = """
    Given the context below, answer the question as clearly and concisely as possible.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retriever = FAISS.load_local(
        folder_path=FAISS_PATH,
        embeddings=embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True  
    ).as_retriever()
    
    qa_chain = RetrievalQA.from_llm(retriever=retriever, llm=load_bart_model(), prompt=prompt)
    return qa_chain


def answer_question(question):
    search_docs = similarity_search(question)
    
    context = " ".join([doc.page_content for doc in search_docs])

    qa_chain = get_qa_chain()
    response = qa_chain.invoke({"query": question, "context": context})
    return response["result"]


