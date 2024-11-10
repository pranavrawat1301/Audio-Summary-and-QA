from audio_to_text import audio_file_to_text
import warnings
from transformers import pipeline, AutoModelForSeq2SeqGeneration, AutoTokenizer
import torch

warnings.filterwarnings("ignore")

# Initialize the summarizer with PyTorch backend explicitly
model_name = "facebook/bart-large-cnn"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name).to(device)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

def response_generator(file_path):
    try:
        searchDocs = audio_file_to_text(file_path)
        
        # Adjust chunk size for token limits
        chunk_size = 500  # Reduced chunk size
        chunks = [searchDocs[i:i+chunk_size] for i in range(0, len(searchDocs), chunk_size)]
        
        summaries = []
        for chunk in chunks:
            if not chunk.strip():  # Skip empty chunks
                continue
                
            try:
                summary = summarizer(chunk, 
                                   max_length=130, 
                                   min_length=30, 
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



