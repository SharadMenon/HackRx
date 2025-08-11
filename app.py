"""
single_file_hackrx_app.py

Single-file FastAPI app for HackRx:
- Uses Ollama (http://localhost:11434) for embeddings and LLM generation
- Uses FAISS for semantic search
- Supports PDF and DOCX document URLs
- Exposes POST /hackrx/run (no authentication required in this version)
- Returns JSON: {"answers": [ ... ]}
"""

import os
import io
import tempfile
import time
from typing import List, Optional
import requests
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from urllib.parse import urlparse
from pathlib import Path

# Document parsing
from docx import Document as DocxDocument
from pypdf import PdfReader

# FAISS
import faiss
import numpy as np

# ------------ Configuration ------------
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")
EMBED_CHUNK_WORDS = 300
TOP_K = 6
REQUEST_TIMEOUT = 30

# -------------- FastAPI app ----------------
app = FastAPI(title="HackRx - Ollama FAISS Q&A (no-auth version)")

# ------------- Request / Response models -------------
class RunRequest(BaseModel, extra="ignore"):
    documents: str
    questions: List[str]
    include_trace: Optional[bool] = False

class RunResponse(BaseModel):
    answers: List[str]

# ------------- Utility functions -------------
def download_to_temp(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    path = urlparse(url).path
    ext = Path(path).suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(resp.content)
    tmp.flush()
    tmp.close()
    return tmp.name

def extract_text_from_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {e}")
    parts = []
    for page in reader.pages:
        try:
            t = page.extract_text()
        except Exception:
            t = ""
        if t:
            parts.append(t)
    return "\n".join(parts)

def extract_text_from_docx(path: str) -> str:
    try:
        doc = DocxDocument(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading DOCX: {e}")
    return "\n".join([p.text for p in doc.paragraphs if p.text and p.text.strip()])

def load_document_text(local_path: str) -> str:
    ext = Path(local_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(local_path)
    elif ext == ".docx":
        return extract_text_from_docx(local_path)
    else:
        try:
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            raise HTTPException(status_code=400, detail=f"Unsupported document type: {ext}")

def simple_chunker(text: str, words_per_chunk: int = EMBED_CHUNK_WORDS) -> List[str]:
    text = text.replace("\r", "\n")
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = []
    current_words = 0
    for p in paragraphs:
        w = p.split()
        if current_words + len(w) <= words_per_chunk:
            current.append(p)
            current_words += len(w)
        else:
            if current:
                chunks.append(" ".join(current))
            if len(w) > words_per_chunk:
                i = 0
                while i < len(w):
                    part = " ".join(w[i:i+words_per_chunk])
                    chunks.append(part)
                    i += words_per_chunk
                current = []
                current_words = 0
            else:
                current = [p]
                current_words = len(w)
    if current:
        chunks.append(" ".join(current))
    merged = []
    buffer = []
    for c in chunks:
        if len(buffer) == 0:
            buffer.append(c)
        else:
            last_words = sum(len(x.split()) for x in buffer)
            if last_words < words_per_chunk * 0.35:
                buffer.append(c)
            else:
                merged.append(" ".join(buffer))
                buffer = [c]
    if buffer:
        merged.append(" ".join(buffer))
    return merged

# -------------- Ollama helpers ----------------
def ollama_embedding(text: str, model: str = EMBED_MODEL, timeout: int = REQUEST_TIMEOUT) -> List[float]:
    url = OLLAMA_API.rstrip("/") + "/api/embeddings"
    payload = {"model": model, "input": text}
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama embedding request failed: {e}")

    if isinstance(data, dict):
        if "embedding" in data and isinstance(data["embedding"], list):
            return data["embedding"]
        if "data" in data and isinstance(data["data"], list) and "embedding" in data["data"][0]:
            return data["data"][0]["embedding"]
        if "embeddings" in data and isinstance(data["embeddings"], list):
            return data["embeddings"][0]
    def find_list(d):
        if isinstance(d, list):
            if d and all(isinstance(x, (int, float)) for x in d):
                return d
            for item in d:
                res = find_list(item)
                if res:
                    return res
        elif isinstance(d, dict):
            for v in d.values():
                res = find_list(v)
                if res:
                    return res
        return None
    vec = find_list(data)
    if vec:
        return vec
    raise HTTPException(status_code=502, detail=f"Unexpected embedding response from Ollama: {data}")

def ollama_generate(prompt: str, model: str = LLM_MODEL, timeout: int = REQUEST_TIMEOUT) -> str:
    url = OLLAMA_API.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.0,
        "stop": None,
        "stream": False
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama generate request failed: {e}")

    if isinstance(data, dict):
        if "response" in data and isinstance(data["response"], str):
            return data["response"]
        if "generated" in data and isinstance(data["generated"], list):
            return "".join(g.get("text", "") for g in data["generated"])
        if "results" in data and isinstance(data["results"], list):
            return "".join(r.get("text", "") for r in data["results"])
        if "choices" in data and isinstance(data["choices"], list):
            return "".join(c.get("text", c.get("message", "")) for c in data["choices"])
    return str(data)

# ------------- Embed store ----------------
class InMemoryEmbedStore:
    def __init__(self):
        self.texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        self.dim: Optional[int] = None

    def index_document(self, text: str):
        chunks = simple_chunker(text)
        vectors = []
        for chunk in chunks:
            emb = ollama_embedding(chunk)
            vectors.append(emb)
            time.sleep(0.01)
        if not vectors:
            raise HTTPException(status_code=400, detail="No text chunks extracted.")
        arr = np.array(vectors).astype("float32")
        self.embeddings = arr
        self.texts = chunks
        self.dim = arr.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(arr)

    def search(self, query: str, top_k: int = TOP_K):
        if self.index is None:
            raise HTTPException(status_code=500, detail="Embeddings not indexed yet.")
        qv = np.array(ollama_embedding(query)).astype("float32").reshape(1, -1)
        D, I = self.index.search(qv, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append({"text": self.texts[idx], "score": float(dist), "index": int(idx)})
        return results

# ------------- Core Q&A pipeline -------------
def build_prompt(context_chunks: List[str], question: str) -> str:
    context_block = "\n\n---\n\n".join(context_chunks)
    return f"""You are an expert insurance and policy analyst. Use ONLY the information in the CONTEXT to answer the QUESTION.
If the answer is not present in the CONTEXT, reply exactly: "Information not found in document".

Format:
- Provide a concise direct answer (1-4 sentences).
- After the answer, include a "Rationale:" line citing chunks.

CONTEXT:
{context_block}

QUESTION:
{question}
"""

def answer_single_question(store: InMemoryEmbedStore, question: str, top_k=TOP_K) -> dict:
    retrieved = store.search(question, top_k=top_k)
    retrieved_sorted = sorted(retrieved, key=lambda x: x["score"])
    context_chunks = [r["text"] for r in retrieved_sorted]
    prompt = build_prompt(context_chunks, question)
    llm_resp = ollama_generate(prompt)
    return {"answer": llm_resp.strip(), "retrieved": retrieved_sorted}

# ------------- API Endpoint -------------
@app.post("/hackrx/run", response_model=RunResponse)
def hackrx_run(payload: RunRequest, Authorization: str = Header(None)):
    local_path = download_to_temp(payload.documents)
    doc_text = load_document_text(local_path)
    if not doc_text.strip():
        raise HTTPException(status_code=400, detail="No text extracted.")
    store = InMemoryEmbedStore()
    store.index_document(doc_text)

    answers_out = []
    for q in payload.questions:
        qa = answer_single_question(store, q, top_k=TOP_K)
        answers_out.append(qa["answer"])

    return {"answers": answers_out}

# ------------- Health / Debug endpoints -------------
@app.get("/health")
def health():
    return {"status": "ok", "ollama": OLLAMA_API, "embed_model": EMBED_MODEL, "llm_model": LLM_MODEL}

@app.get("/")
def root():
    return {"msg": "HackRx Ollama FAISS single-file app (no auth). POST /hackrx/run to use."}
