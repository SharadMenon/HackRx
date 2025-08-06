# 3_retrieve_clauses.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

# Always ensure 'punkt' is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load from the local folder instead of downloading
model = SentenceTransformer('./local_model')


def load_faiss_index(index_path="faiss_index.idx", metadata_path="metadata.pkl"):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        texts, metadata = pickle.load(f)
    return index, texts, metadata

def build_semantic_query(parsed_query):
    # Combine procedure, location, duration into a meaningful sentence
    procedure = parsed_query.get("procedure", "")
    location = parsed_query.get("location", "")
    duration = parsed_query.get("policy_duration", "")
    return f"Policy coverage for {procedure} in {location} with policy active for {duration}"

from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

model = SentenceTransformer('./local_model') # This is already downloaded

def retrieve_relevant_clauses(parsed_query, top_k=5, top_sentences=2):
    from embed_pdf import load_faiss_index

    index, docs = load_faiss_index("index")
    query_text = f"{parsed_query['procedure']} {parsed_query['location']} {parsed_query['policy_duration']} {parsed_query['age']} {parsed_query['gender']}"

    query_embedding = model.encode(query_text, convert_to_tensor=True)
    scores, indices = index.search(query_embedding.unsqueeze(0).cpu().numpy(), top_k)
    
    results = []

    for idx in indices[0]:
        doc = docs[idx]
        source = doc['source']
        chunk = doc['text']

        # Split chunk into sentences
        sentences = sent_tokenize(chunk)
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
        top_sentence_ids = similarities.topk(top_sentences).indices.tolist()

        matched_sentences = [sentences[i] for i in top_sentence_ids]
        highlighted_text = " ".join(matched_sentences)

        results.append({
            "clause": highlighted_text,
            "source": source
        })

    return results

def index_and_retrieve(pdf_path, parsed_query, top_k=5):
    # Import inside to avoid circular dependencies
    from embed_pdf import extract_pdf_text, chunk_text, compute_embeddings, save_faiss_index, load_faiss_index
    import os

    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text)
    embeddings = compute_embeddings(chunks)
    
    # Save and load FAISS
    save_faiss_index(embeddings, chunks, "index")
    index, docs = load_faiss_index("index")

    return retrieve_relevant_clauses(parsed_query, top_k=top_k)

# Example usage
if __name__ == "__main__":
    from pprint import pprint
    parsed_query = {
        "age": 46,
        "gender": "male",
        "procedure": "knee surgery",
        "location": "Pune",
        "policy_duration": "3 months"
    }
    top_clauses = retrieve_relevant_clauses(parsed_query)
    pprint(top_clauses)
