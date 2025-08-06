import streamlit as st
import os
import tempfile
from parse_query import parse_query_natural_language
from decide import build_decision_prompt, ask_ollama

st.set_page_config(page_title="Policy Claim LLM Assistant", layout="centered")

st.title("🧠 Policy Query Decision Assistant (Local + Free)")

st.markdown("""
Upload a policy document (PDF), type your query, and let the local AI determine the decision with justification.
""")

# --- File Upload ---
uploaded_file = st.file_uploader("📄 Upload Insurance Policy PDF", type=["pdf"])
query = st.text_input("🔍 Enter your query", placeholder="e.g., 46-year-old male, knee surgery in Pune, 3-month-old policy")
if uploaded_file and query:
    from retrieve_clauses import index_and_retrieve
if uploaded_file and query:
    with st.spinner("📚 Reading and indexing policy document..."):
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # Retrieve top clauses
        parsed_query = parse_query_natural_language(query)
        top_clauses = index_and_retrieve(pdf_path, parsed_query, top_k=5)

    with st.spinner("🤖 Asking local AI model for a decision..."):
        prompt = build_decision_prompt(query, parsed_query, top_clauses)
        response = ask_ollama(prompt)

    st.subheader("📤 Structured Response")
    st.json(response)

    st.subheader("📎 Matched Clauses")
    for i, clause in enumerate(top_clauses, 1):
        with st.expander(f"Clause {i} - from {os.path.basename(clause['source'])}"):
            st.markdown(f"```\n{clause['clause']}\n```")
else:
    st.info("Upload a PDF and enter your query to begin.")
