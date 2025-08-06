# 4_decide.py

import subprocess
import json

def ask_ollama(prompt, model="mistral"):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8")

def build_decision_prompt(user_query, parsed_query, retrieved_clauses):
    prompt = f"""
You are an insurance claim decision assistant.

User Query:
"{user_query}"

Parsed Details:
{json.dumps(parsed_query, indent=2)}

Relevant Clauses:
"""
    for i, clause in enumerate(retrieved_clauses, 1):
        prompt += f"\nClause {i} (from {clause['source']}): {clause['clause']}\n"

    prompt += """
---
Based on the clauses and the structured query details:
1. Decide whether the claim should be APPROVED or REJECTED.
2. Mention payout amount, if any.
3. Justify the decision by clearly referencing clause numbers or content.

Respond ONLY in JSON like this:
{
  "decision": "Approved",
  "amount": "₹50,000",
  "justification": "Clause 2 states that surgeries like knee replacement are covered after 90 days. The user has completed 3 months of policy.",
  "matched_clauses": ["Clause 2"]
}
"""
    return prompt

# Example usage
if __name__ == "__main__":
    from pprint import pprint
    from parse_query import parse_query_natural_language
    from retrieve_clauses import retrieve_relevant_clauses

    query = "46-year-old male, knee surgery in Pune, 3-month-old policy"
    parsed = parse_query_natural_language(query)
    clauses = retrieve_relevant_clauses(parsed, top_k=5)

    prompt = build_decision_prompt(query, parsed, clauses)
    response = ask_ollama(prompt)
    
    try:
        decision_json = json.loads(response.strip().split("```")[-1] if "```" in response else response)
        pprint(decision_json)
    except Exception:
        print("⚠️ Could not parse JSON, here is raw output:")
        print(response)
