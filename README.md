# LLM-based Natural Language Query Understanding & Clause-Based Decision System

## 📌 Overview
This project is a **document Q&A system** with:
- **Semantic search** on uploaded documents (PDFs)
- **Clause-based decision-making**
- Powered by a **local LLM (Mistral)** via Ollama
- **FastAPI** backend with optional public exposure via ngrok

It allows users to upload documents, query them in natural language, and receive **structured JSON answers** with relevant clauses and justification.

---

## 🚀 Features
- **Local inference** (no API keys required)
- **Semantic clause retrieval** using FAISS
- **Natural language understanding** via Mistral
- **Decision logic** for YES/NO style answers
- **FastAPI server** with endpoints for:
  - Document upload & embedding
  - Query answering
  - Clause-based decision making
- **Optional public link** using ngrok for demo purposes

---


## 🛠️ Installation

### 1️⃣ Install Python & Dependencies
Make sure you have **Python 3.10+** installed.


# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install fastapi uvicorn pypdf2 faiss-cpu
pip install ollama
do ollama pull mistral
and ollama pull nomic-embed-text
Now,
Run the backend server:
uvicorn app_final_with_auth_fixed:app --reload --host 0.0.0.0 --port 8000
The API will be available at:
http://127.0.0.1:8000

If you want to share your API with others during a hackathon/demo:
ngrok.exe http 8000

