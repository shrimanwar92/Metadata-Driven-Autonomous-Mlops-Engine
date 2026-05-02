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
    Analyze the following schema and domain to create a technical Feature Engineering Strategy for Clustering.

    DOMAIN: {domain_name}
    SCHEMA (Cleaned Data Profile):
    {schema_str}

    CRITICAL INSTRUCTIONS FOR SUBJECT_ID:
    1. Identify the 'subject_id' (the entity we are clustering). 
    2. ANTI-LEAKAGE RULE: If a column has a 1:1 relationship with rows or is perfectly correlated with the index (e.g., 'generated_id', 'Unnamed: 0'), it is NOT a subject_id. It is a technical index that will cause 100% training leakage. 
    3. If no physical entity anchor (like JunctionID or StationID) exists, you MUST set "subject_id": null. Do not hallucinate a primary key as an entity.

    TASK:
    1. Define 'Logical Interactions': Create behavioral signals (ratios/multiplication). Specify outlier clipping (e.g., 99th percentile) and zero-fill strategies for division.[cite: 3]
    2. Preprocessing Policy: Choose a scaler (PowerTransformer for skewed data, RobustScaler for outliers) and an imputation strategy.[cite: 3, 6]
    3. Aggregation Strategy: Define how to group data by the validated subject_id. Specify which columns should be 'mean', 'sum', or 'mode' (for categorical context).[cite: 5]
    4. Feature Selection: Set thresholds for max correlation and variance to prevent redundancy.
    5. Dimensionality Constraint: If the number of generated features exceeds 15, recommend a PCA variance retention threshold (e.g., 95%) in the metadata.

    OUTPUT FORMAT:
    Return ONLY a valid JSON object:
    {{
        "interaction_priorities": [
            {{
                "pair": ["col1", "col2"], 
                "logic": "multiplication" | "ratio", 
                "name": "meaningful_name",
                "outlier_clipping_percentile": 99,
                "error_handling": "zero_fill"
            }}
        ],
        "preprocessing_metadata": {{
            "recommended_scaler": "PowerTransformer" | "RobustScaler",
            "imputation_strategy": "median" | "mean",
            "categorical_encoding": "one-hot"
        }},
        "subject_id": "string_or_null",
        "aggregation_strategy": {{
            "columns_to_sum": [],
            "columns_to_mean": [],
            "columns_to_mode": []
        }},
        "feature_selection": {{
            "max_correlation_threshold": 0.85,
            "variance_threshold": 0.01
        }},
        "recommended_algorithm": "GMM" | "KMeans"
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