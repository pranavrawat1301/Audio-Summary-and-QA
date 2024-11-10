import streamlit as st
import os
import tempfile
from audio_to_text import audio_file_to_text
from summary import response_generator
from work_on_text import answer_question
from load_transcript import save_documents_to_faiss

def main():
    st.set_page_config(page_title="Audio Summarizer & Q&A", layout="wide")
    
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📝 Audio Summarizer & Q&A System")
    
    # File upload section
    st.header("Upload Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=['mp3'])
    
    if uploaded_file:
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        # Process button
        if st.button("Process Audio"):
            with st.spinner('Transcribing audio...'):
                # Save to FAISS and get transcript
                save_documents_to_faiss(temp_file_path)
                st.success("Audio processed successfully!")
            
            with st.spinner('Generating summary...'):
                # Generate summary
                summary = response_generator(temp_file_path)
                st.session_state['summary'] = summary
                st.session_state['file_processed'] = True
        
        # Display summary if available
        if 'summary' in st.session_state:
            st.header("Summary")
            st.write(st.session_state['summary'])
        
        # Q&A section
        if 'file_processed' in st.session_state:
            st.header("Ask Questions")
            question = st.text_input("Enter your question about the audio:")
            
            if st.button("Get Answer"):
                if question:
                    with st.spinner('Finding answer...'):
                        answer = answer_question(question)
                        st.subheader("Answer:")
                        st.write(answer)
                else:
                    st.warning("Please enter a question.")
        
        # Cleanup temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass
    
    # Instructions
    else:
        st.info("Please upload an MP3 file to get started.")
        
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

if __name__ == "__main__":
    main()