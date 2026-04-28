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

def generate_domain_policy(max_retries=5):
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

    # 3. Prompt Configuration
    system_instruction = "You are a specialized Data Architect API. Output ONLY raw JSON. No markdown, no preamble."
    
    prompt = f"""
    Act as an Expert Data Architect and Behavioral Scientist. 
    Analyze the following schema and categorize the columns for an Unsupervised Clustering (Persona) task.

    SCHEMA:
    {schema_str}

    TASK:
    1. Identify the 'domain' (e.g., Clinical, Financial, Traffic, Retail).
    2. Identify the 'subject_id': 
       - This must be the most granular entity anchor (e.g., CustomerID, PatientID, InvoiceNo).
       - If no explicit ID exists, return null.
    3. Identify 'protected_features': 
       - These are "High-Signal" behavioral columns critical for clustering in this domain.
       - Examples: 'Spending Score' for Retail, 'heart_rate' for Clinical, 'flow_rate' for Traffic.
    4. Identify 'technical_garbage': 
       - Columns with zero entropy for clustering (UUIDs, internal timestamps, row indexes).
    5. Logic for 'Drop vs. Keep':
       - If a column is a redundant string (like "Name" or "Email"), suggest it for dropping.
       - If a column represents a category (like "Gender" or "Region"), keep it for encoding.

    OUTPUT FORMAT:
    Return ONLY a valid JSON object:
    {{
        "domain": "string",
        "subject_id": "string or null",
        "protected_features": ["list", "of", "strings"],
        "technical_garbage": ["list", "of", "strings"]
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

            # 5. Parse and Save Policy
            policy = json.loads(response.text)
            
            with open(DOMAIN_POLICY_PATH, 'w') as f:
                json.dump(policy, f, indent=4)
            
            print(f"✅ Domain Policy Saved: {policy.get('domain', 'Unknown').upper()} discovered.")
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

if __name__ == "__main__":
    generate_domain_policy()