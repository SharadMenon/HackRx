import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')  # Ensure fresh download

text = "Hello. How are you? I'm fine."
print(sent_tokenize(text))
