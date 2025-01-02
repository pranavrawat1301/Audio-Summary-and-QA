import assemblyai as aai


TRANSCRIPTION_PATH = r"D:\Projects(internship)\LLM based Projects\Audio Summarizer\transcription.txt"
FAISS_PATH = r"D:\f-ai\Pdf vocal query bot\faiss_local"
INDEX_NAME = "faiss_index"

def audio_file_to_text(file_path):
    aai.settings.api_key = "YOUR API KEY"
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    
    with open(TRANSCRIPTION_PATH, "w") as f:
        f.write(transcript.text)
    
    return transcript.text
