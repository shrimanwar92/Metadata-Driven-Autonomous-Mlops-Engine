import json
import os
import time
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Add parent directory to path to import constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import PROFILER_CLEAN_REPORT_PATH, DOMAIN_POLICY_PATH, FEATURE_ENGG_STRATEGY_PATH

load_dotenv()

def generate_feature_engg_strategy(max_retries=5):
    """
    Consults Gemini to create a Semantic Feature Engineering Strategy.
    This ensures that interaction terms (X*Y) and ratios (X/Y) have 
    logical domain meaning.
    """
    # 1. Initialize Gemini Client
    api_key = os.getenv("MDAME_API_KEY")
    if not api_key:
        raise ValueError("MDAME_API_KEY not found in environment variables.")
        
    client = genai.Client(api_key=api_key)
    
    # 2. Load existing context (Cleaned Report and Domain Policy)
    if not os.path.exists(PROFILER_CLEAN_REPORT_PATH):
        raise FileNotFoundError(f"Cleaned report not found at {PROFILER_CLEAN_REPORT_PATH}")
    
    with open(PROFILER_CLEAN_REPORT_PATH, 'r') as f:
        report = json.load(f)
    with open(DOMAIN_POLICY_PATH, 'r') as f:
        policy = json.load(f)

    variables = report.get('variables', {})
    domain_name = policy.get("domain", "Unknown")
    protected = policy.get("protected_features", [])
    
    # Extract schema specifically from the cleaned Silver layer
    schema_info = [f"- {col} ({details.get('type', 'Unknown')})" for col, details in variables.items()]
    schema_str = "\n".join(schema_info)

    # 3. Prompt Configuration (The Semantic Brain)
    system_instruction = "You are a Senior Feature Engineer API. Output ONLY raw JSON."
    
    prompt = f"""
    Act as a Senior Data Scientist and Domain Expert. 
    Analyze the following schema and domain to create a Feature Engineering Strategy for Clustering.

    DOMAIN: {domain_name}
    PROTECTED FEATURES: {protected}
    SCHEMA (Cleaned Data):
    {schema_str}

    TASK:
    1. Identify 'Logical Interactions': Which numerical columns should be multiplied/divided to create high-value behavioral signals? (e.g., 'UnitPrice * Quantity' = Revenue).
    2. Identify 'Semantic Groupings': Group features by intent (e.g., 'Spending Habits', 'Demographics', 'Traffic Flow').
    3. Propose 'New Dimensions': Suggest 3 synthetic features that define personas in this domain.
    4. Define 'Pruning Priority': If features are highly correlated, which one is more semantically important to keep?

    OUTPUT FORMAT:
    Return ONLY a valid JSON object:
    {{
        "interaction_priorities": [
            {{"pair": ["col1", "col2"], "logic": "multiplication", "name": "meaningful_name"}},
            {{"pair": ["col1", "col2"], "logic": "ratio", "name": "meaningful_ratio"}}
        ],
        "semantic_groups": {{
            "group_name": ["list_of_cols"]
        }},
        "synthetic_targets": ["dimension_description_1", "dimension_description_2"]
    }}
    """

    # 4. API Call with Exponential Backoff
    for attempt in range(max_retries):
        try:
            print(f"🧠 Generating Semantic Strategy (Attempt {attempt + 1}/{max_retries})...")
            
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    temperature=0.2 
                ),
                contents=prompt
            )

            # 5. Parse and Save Strategy
            strategy = json.loads(response.text)
            
            with open(FEATURE_ENGG_STRATEGY_PATH, 'w') as f:
                json.dump(strategy, f, indent=4)
            
            print(f"✅ Feature Engineering Strategy Saved for {domain_name.upper()}.")
            return strategy

        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                wait_time = (2 ** attempt) + 5 
                print(f"⚠️ Rate limit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ Error: {e}")
                raise e

    raise Exception("Max retries exceeded for Feature Engineering Strategy.")

if __name__ == "__main__":
    generate_feature_engg_strategy()