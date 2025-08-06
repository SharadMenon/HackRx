from sentence_transformers import SentenceTransformer

# Load and download the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Save it locally
model.save('./local_model')
print("Model downloaded and saved to ./local_model")
