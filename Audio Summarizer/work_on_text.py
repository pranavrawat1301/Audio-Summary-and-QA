from load_transcript import load_bart_model, get_embeddings
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

@st.cache_data
def similarity_search(user_question, db):
    """Cache similarity search results"""
    search_docs = db.similarity_search(
        user_question,
        k=2  # Reduce number of documents retrieved
    )
    return search_docs

@st.cache_resource
def get_qa_chain():
    """Cache QA chain creation"""
    prompt_template = """
    Answer the question concisely based on the context below.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    embeddings = get_embeddings()
    retriever = FAISS.load_local(
        folder_path=FAISS_PATH,
        embeddings=embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True  
    ).as_retriever(
        search_kwargs={"k": 2}  # Reduce number of documents retrieved
    )
    
    qa_chain = RetrievalQA.from_llm(
        retriever=retriever,
        llm=load_bart_model(),
        prompt=prompt
    )
    return qa_chain

def answer_question(question, db=None):
    """Optimized question answering"""
    search_docs = similarity_search(question, db) if db else []
    context = " ".join([doc.page_content for doc in search_docs])
    
    qa_chain = get_qa_chain()
    response = qa_chain.invoke({
        "query": question,
        "context": context[:1024]  # Limit context length for faster processing
    })
    return response["result"]


