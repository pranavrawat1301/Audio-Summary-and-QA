### Audio Summarizer and Q&A System

This repository provides a comprehensive solution for processing audio files into textual transcriptions, summarizing their content, and enabling question-and-answer functionality based on the processed data. The system leverages state-of-the-art natural language processing models, vector databases, and a user-friendly interface for seamless interactions.

---

### **Features**
1. **Audio Transcription**  
   - Converts audio files (MP3 format) into textual data using the AssemblyAI API.
   
2. **Summarization**  
   - Summarizes lengthy transcriptions into concise and meaningful summaries using a pre-trained BART model from Hugging Face.

3. **Query and Retrieval**  
   - Allows users to query the transcribed content, retrieving contextually relevant answers using FAISS (Facebook AI Similarity Search) for efficient document indexing and retrieval.

4. **Streamlit-based Interface**  
   - An intuitive web application built with Streamlit for uploading audio files, viewing summaries, and interacting with the Q&A system.

---

### **Tech Stack**
- **Python**: Core programming language for processing and logic.
- **AssemblyAI**: Used for high-quality audio transcription.
- **Hugging Face Transformers**: Pre-trained BART model for summarization and question-answering pipeline.
- **FAISS**: Efficient similarity search and indexing for handling large text datasets.
- **Streamlit**: Provides an interactive web-based user interface.

---

### **Project Structure**
- **`audio_to_text.py`**  
   Handles audio file transcription using AssemblyAI and saves the transcription to a file.

- **`load_transcript.py`**  
   Loads the transcription, splits it into chunks, embeds it, and stores it in a FAISS index.

- **`summary.py`**  
   Summarizes transcriptions into concise content using the BART model.

- **`work_on_text.py`**  
   Implements a similarity search and retrieval-based question-answering system using FAISS and Hugging Face pipelines.

- **`save.py`**  
   A utility script to preprocess and save audio data into FAISS for future queries.

- **`main.py`**  
   A command-line interface to interact with the system by generating summaries or querying the audio content.

- **`streamlit_app.py`**  
   Provides a web-based interface for audio processing, summary generation, and question answering.

---

### **Usage**
1. **Setup and Installation**  
   - Clone the repository.  
   - Install dependencies using `pip install -r requirements.txt`.  
   - Add your AssemblyAI API key to `audio_to_text.py`.  

2. **Run Command-Line Interface**  
   - Execute `main.py` for CLI-based interaction.

3. **Run Streamlit Web App**  
   - Run `streamlit_app.py` using `streamlit run streamlit_app.py`.  
   - Upload an MP3 file, generate summaries, and ask questions through the app.

---

### **Key Dependencies**
- **AssemblyAI**: For transcription.
- **Transformers**: For text summarization and Q&A pipelines.
- **FAISS**: For similarity search.
- **Streamlit**: For UI.

---

### **Future Improvements**
- Add support for additional audio formats.
- Enhance summarization using newer models like ChatGPT or similar.
- Incorporate multilingual support for non-English audio.
- Improve indexing and retrieval efficiency for very large datasets.

---

### **Contributions**
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

### **License**
This project is licensed under the MIT License. See the LICENSE file for more details. 

---

### **Acknowledgements**
- AssemblyAI for robust transcription services.
- Hugging Face for their state-of-the-art NLP models.
- FAISS for efficient similarity search solutions.
- Streamlit for creating interactive web apps.


---

### **Some Screenshots of the StreamLit App**


**Landing Page**
![ss3](https://github.com/user-attachments/assets/e89eb2f5-cbb3-4e38-b432-f2ed7ae9e33d)


**Summary**
![ss1 (1)](https://github.com/user-attachments/assets/726ec687-2ed9-4753-ab34-9e07581a34c0)


**Question-Answering**
![ss2](https://github.com/user-attachments/assets/fd6d24de-3acc-458c-be15-b661fcdb2c0e)
