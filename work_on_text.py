import os
import warnings
from transformers import pipeline, AutoModelForSeq2SeqGeneration, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# Suppress warnings
warnings.filterwarnings("ignore")

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
        device=-1  # CPU, use 0 for GPU if available
    )
    return HuggingFacePipeline(pipeline=pipe)

def similarity_search(user_question):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    db = FAISS.load_local(
        folder_path=FAISS_PATH,
        embeddings=embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True
    )
    
    return db.similarity_search(user_question)

def get_qa_chain():
    prompt_template = """
    Answer the question based on the context below:
    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    retriever = FAISS.load_local(
        folder_path=FAISS_PATH,
        embeddings=embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True
    ).as_retriever()
    
    return RetrievalQA.from_llm(
        llm=load_bart_model(),
        retriever=retriever,
        prompt=prompt
    )

def answer_question(question):
    try:
        search_docs = similarity_search(question)
        context = " ".join([doc.page_content for doc in search_docs])
        qa_chain = get_qa_chain()
        response = qa_chain.invoke({"query": question})
        return response["result"]
    except Exception as e:
        return f"Error generating answer: {str(e)}"


