from audio_to_text import audio_file_to_text
import torch
from transformers import pipeline, AutoModelForSeq2SeqGeneration, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

# Initialize the summarizer with PyTorch backend explicitly
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="pt")

def response_generator(file_path):
    try:
        searchDocs = audio_file_to_text(file_path)
        
        # Adjust chunk size to avoid token length issues
        chunk_size = 1000  # Reduced from 1024 to allow for some padding
        chunks = [searchDocs[i:i+chunk_size] for i in range(0, len(searchDocs), chunk_size)]
        
        summaries = []
        for chunk in chunks:
            if not chunk.strip():  # Skip empty chunks
                continue
                
            try:
                summary = summarizer(chunk, 
                                   max_length=150, 
                                   min_length=50, 
                                   do_sample=False,
                                   truncation=True)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
        
        if not summaries:
            return "Could not generate summary. Please try with different content."
        
        return " ".join(summaries)
    except Exception as e:
        print(f"Error in response generation: {str(e)}")
        return f"An error occurred while generating the summary: {str(e)}"



