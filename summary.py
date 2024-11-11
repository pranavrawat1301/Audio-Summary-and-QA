from audio_to_text import audio_file_to_text
from transformers import pipeline


import warnings
import tensorflow as tf

# Suppress warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

summarizer = pipeline("summarization")

def response_generator(file_path):
    searchDocs = audio_file_to_text(file_path)
    chunk_size = 1024  # Ensure each chunk is within the limit
    chunks = [searchDocs[i:i+chunk_size] for i in range(0, len(searchDocs), chunk_size)]
    
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    return " ".join(summaries)

#from concurrent.futures import ThreadPoolExecutor, as_completed

'''def response_generator(file_path):
    searchDocs = audio_file_to_text(file_path)
    chunk_size = 1024
    chunks = [searchDocs[i:i+chunk_size] for i in range(0, len(searchDocs), chunk_size)]
    
    summaries = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(summarizer, chunk, max_length=150, min_length=50, do_sample=False) for chunk in chunks]
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary[0]['summary_text'])
    
    return " ".join(summaries)'''



