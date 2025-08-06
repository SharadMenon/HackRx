import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
model = SentenceTransformer('./local_model')

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, source="your_pdf.pdf", chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append({"text": chunk, "source": source})
        start += chunk_size - overlap
    return chunks


def compute_embeddings(chunks):
    embeddings = model.encode([chunk["text"] for chunk in chunks], convert_to_tensor=False)
    return embeddings

def save_faiss_index(embeddings, chunks, index_path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, f"{index_path}.index")
    with open(index_path + "_docs.json", "w", encoding="utf-8") as f:
        json.dump([{"text": chunk, "source": f"Clause {i+1}"} for i, chunk in enumerate(chunks)], f)




def load_faiss_index(index_path):
    index = faiss.read_index(f"{index_path}.index")
    with open(index_path + "_docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    return index, docs
