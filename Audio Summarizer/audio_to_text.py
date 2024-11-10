import assemblyai as aai
import os

# Use a relative path
TRANSCRIPTION_PATH = os.path.join(os.getcwd(), "transcription.txt")
FAISS_PATH = os.path.join(os.getcwd(), "faiss_local")
INDEX_NAME = "faiss_index"

def audio_file_to_text(file_path):
    aai.settings.api_key = "b7c7f111370849ed8f2256ab63364594"
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    
    with open(TRANSCRIPTION_PATH, "w") as f:
        f.write(transcript.text)
    
    return transcript.text
