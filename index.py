import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer('./local_model')

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def index_documents(folder_path):
    texts = []
    metadata = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(folder_path, filename))
            chunks = chunk_text(text)
            for chunk in chunks:
                texts.append(chunk)
                metadata.append({"source": filename})
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    faiss.write_index(index, "faiss_index.idx")
    with open("metadata.pkl", "wb") as f:
        pickle.dump((texts, metadata), f)
    print(f"Indexed {len(texts)} text chunks.")


index_documents("docs") 
