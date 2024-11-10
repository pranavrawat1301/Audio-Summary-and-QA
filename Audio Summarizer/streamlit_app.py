import streamlit as st
import os
import tempfile
from audio_to_text import audio_file_to_text
from summary import response_generator
from work_on_text import answer_question
from load_transcript import save_documents_to_faiss
import torch

# Configure page
st.set_page_config(
    page_title="Audio Summarizer & Q&A",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

if 'db' not in st.session_state:
    st.session_state.db = None

def process_audio(temp_file_path):
    """Process audio file with progress tracking"""
    with st.spinner('Transcribing audio...'):
        transcript = audio_file_to_text(temp_file_path)
    
    with st.spinner('Processing transcript...'):
        # Save to FAISS and cache the database
        st.session_state.db = save_documents_to_faiss(temp_file_path, transcript)
        st.session_state.processed_files.add(temp_file_path)
    
    with st.spinner('Generating summary...'):
        summary = response_generator(temp_file_path)
        st.session_state['summary'] = summary
        st.session_state['file_processed'] = True

def main():
    st.title("📝 Audio Summarizer & Q&A System")
    
    # Display GPU status
    device = "GPU ✨" if torch.cuda.is_available() else "CPU 💻"
    st.sidebar.info(f"Running on: {device}")
    
    # File upload section
    uploaded_file = st.file_uploader("Choose an audio file", type=['mp3'])
    
    if uploaded_file:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        # Process button
        if temp_file_path not in st.session_state.processed_files:
            if st.button("Process Audio"):
                process_audio(temp_file_path)
        
        # Display summary
        if 'summary' in st.session_state:
            st.header("Summary")
            st.write(st.session_state['summary'])
        
        # Q&A section
        if 'file_processed' in st.session_state:
            st.header("Ask Questions")
            question = st.text_input("Enter your question about the audio:")
            
            if question:
                if st.button("Get Answer"):
                    with st.spinner('Finding answer...'):
                        answer = answer_question(question, st.session_state.db)
                        st.subheader("Answer:")
                        st.write(answer)
        
        # Cleanup
        try:
            os.unlink(temp_file_path)
        except:
            pass
    
    else:
        st.info("Please upload an MP3 file to get started.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
