# 2_parse_query.py
import subprocess
import json

def ask_ollama_mistral(prompt, model="mistral"):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8")

def parse_query_natural_language(nl_query):
    system_prompt = """
You are an assistant that extracts structured fields from vague insurance-related queries.
Given a user query, extract the following fields in JSON format:
- age (integer)
- gender (male/female/other)
- procedure (medical procedure or condition)
- location (city or state)
- policy_duration (e.g., '3 months', '6 years')

Only return the JSON.
"""

    full_prompt = f"{system_prompt}\n\nQuery: {nl_query}"
    output = ask_ollama_mistral(full_prompt)
    
    # Try to extract JSON from messy output
    try:
        json_data = json.loads(output.strip().split("```")[-1] if "```" in output else output)
        return json_data
    except Exception as e:
        print("⚠️ Could not parse JSON. Raw output:")
        print(output)
        return {}

# Example usage
query = "46-year-old male, knee surgery in Pune, 3-month-old policy"
parsed = parse_query_natural_language(query)
print(parsed)
