import streamlit as st
from audio_to_text import audio_file_to_text
from summary import response_generator
from work_on_text import answer_question
import os

# Streamlit app definition
def main():
    st.title("Audio Summarizer and Interactive Q&A System")
    st.write("Upload an audio file to get a summary and ask questions about it.")

    # File uploader for audio files
    audio_file = st.file_uploader("Upload your audio file (MP3 format)", type=["mp3"])

    if audio_file is not None:
        # Save uploaded file temporarily
        temp_file_path = os.path.join("temp_audio.mp3")
        with open(temp_file_path, "wb") as f:
            f.write(audio_file.read())

        # Generate and display summary
        st.header("Audio Summary")
        with st.spinner("Transcribing and summarizing the audio..."):
            transcript_text = audio_file_to_text(temp_file_path)
            summary = response_generator(temp_file_path)
        st.success("Summary generated!")
        st.write(summary)

        # Interactive Q&A section
        st.header("Ask a Question")
        question = st.text_input("Type your question about the audio content here:")
        if question:
            with st.spinner("Retrieving the answer..."):
                answer = answer_question(question)
            st.success("Answer retrieved!")
            st.write("**Answer**: ", answer)

if __name__ == "__main__":
    main()
