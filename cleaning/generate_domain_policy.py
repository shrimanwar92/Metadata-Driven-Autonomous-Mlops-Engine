import json
import os
import time
import sys
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv

# Add parent directory to path to import constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import PROFILER_REPORT_PATH, DOMAIN_POLICY_PATH

load_dotenv()

def get_domain_policy_from_llm(max_retries=5):
    """
    Generates a domain policy using Gemini 1.5 Flash with exponential backoff 
    to mitigate rate limiting.
    """
    # 1. Initialize Gemini Client
    api_key = os.getenv("MDAME_API_KEY")
    
    if not api_key:
        raise ValueError("MDAME_API_KEY not found in environment variables.")
    client = genai.Client(api_key=api_key)
    
    # 2. Extract schema from Profiler Report
    if not os.path.exists(PROFILER_REPORT_PATH):
        raise FileNotFoundError(f"Profiler report not found at {PROFILER_REPORT_PATH}")

    with open(PROFILER_REPORT_PATH, 'r') as f:
        report = json.load(f)
    
    variables = report.get('variables', {})
    schema_info = [f"- {col} ({details.get('type', 'Unknown')})" for col, details in variables.items()]
    schema_str = "\n".join(schema_info)

    vars_summary = {}
    for col, stats in report.get('variables', {}).items():
        vars_summary[col] = {
            "type": stats.get('type'),
            "distinct_percent": stats.get('p_distinct'),
            "missing_percent": stats.get('p_missing'),
            "is_unique": stats.get('is_unique'),
            "skewness": stats.get('skewness')
        }

    # 3. Prompt Configuration
    system_instruction = "You are a specialized Data Architect API. Output ONLY raw JSON. No markdown, no preamble."
    
    prompt = f"""
    Act as an Expert Data Architect. Analyze the schema below for Unsupervised Clustering.
    Analyze this data profile and generate a Domain Policy: {json.dumps(vars_summary)}.

    Consider high-level statistical nuances:
    - Identify the 'subject_id' (e.g., CustomerID). A column name containing 'id' or 'Id' or 'ID' string.
    - If the column name contains id and is aggregatable is important column.
    - If a column name contains id and is a 'Unique Identifier' (unique_values == total_rows or contains 95% unique values), it is **Technical Garbage** (e.g., TransactionID).
    - Identify 'protected_features': High-variance behavioral metrics critical for personas.
    - Identify 'technical_garbage': Metadata, sequence IDs, and low-entropy strings.
    
    OUTPUT ONLY VALID JSON:
    {{
        "domain": "string",
        "subject_id": "string or null",
        "protected_features": [],
        "technical_garbage": []
    }}
    """

    # 4. API Call with Exponential Backoff
    for attempt in range(max_retries):
        try:
            print(f"🧠 Consulting Gemini (Attempt {attempt + 1}/{max_retries})...")
            
            response = client.models.generate_content(
                model="gemini-2.5-flash", # Higher rate limits than Pro
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    temperature=0.1 # Low temp for consistent structural output
                ),
                contents=prompt
            )

            # 5. Parse and return Policy
            policy = json.loads(response.text)
            return policy

        except Exception as e:
            # Check for rate limit (429) or overloaded (503)
            if "429" in str(e) or "503" in str(e) or "ResourceExhausted" in str(e):
                wait_time = (2 ** attempt) + 2 # 2, 4, 8, 16...
                print(f"⚠️ Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ Unrecoverable Error: {e}")
                raise e

    raise Exception("Max retries exceeded. Please check your AI Studio quota.")

def generate_domain_policy():
    if os.path.exists(DOMAIN_POLICY_PATH):
        with open(DOMAIN_POLICY_PATH, "r") as f:
            policy = json.load(f)
    else:
        policy = get_domain_policy_from_llm()

    with open(DOMAIN_POLICY_PATH, 'w') as f:
        json.dump(policy, f, indent=4)
    
    print(f"✅ Domain Policy Saved: {policy.get('domain', 'Unknown').upper()} discovered.")
        
    

if __name__ == "__main__":
    generate_domain_policy()