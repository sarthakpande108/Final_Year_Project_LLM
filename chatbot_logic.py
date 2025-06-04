# chatbot_logic.py

import os
import json
import re
from concurrent.futures import ThreadPoolExecutor
from fetch_risk_utils import fetch_user_data, calculate_rule_based_risk, parse_json_field
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
load_dotenv()


# --- 1. Imports & Gemini Configuration ---
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
client = genai.GenerativeModel(model_name="gemini-1.5-flash")
  # Replace with your actual API base URL
# --- 2. Load & Cache Your Data ---
def load_data(
    stock_path='nifty100_seqdata.txt',
    mf_path='mutual_fund_sequential_data.txt',
    loan_path='loan_sequential_data.txt'
):
    stocks = [json.loads(line) for line in open(stock_path, 'r')]
    mf_blocks = open(mf_path, 'r', encoding='utf-8').read().strip().split("\n\n")
    loan_docs = open(loan_path, 'r', encoding='utf-8').read().strip().split("\n\n")
    print(f"🌟 Loaded {len(stocks)} stocks, {len(mf_blocks)} mutual funds, {len(loan_docs)} loan entries.")
    return stocks, mf_blocks, loan_docs

stocks, mf_blocks, loan_docs = load_data()

# --- 3. Few-Shot Prompt Classifier ---
def classify_query(query: str) -> list[str]:
    few_shot = f"""
You are a router that reads a user’s financial query and returns one or more keywords from:
- STOCK
- MUTUAL_FUND
- LOAN
- BOTH
-FINANICAL_QUERY

Examples:
Q: "Which stock has the highest dividend yield?"  → STOCK
Q: "Suggest top flexi-cap mutual funds for medium risk."  → MUTUAL_FUND
Q: "Which bank provides the lowest home loan EMI?"  → LOAN
Q: "I want 1000Rs/month in large-cap funds and car loan options." → BOTH (STOCK,MUTUAL_FUND,LOAN)
Q: "HOW CAN I ACHEIVE MY GOAL,OR SUGGEST ME SOME EARLY LOAN PAYMENT STRATEGY?"  → FINAICAL_QUERY

Now classify:
Q: "{query}"
A:"""

    resp = client.generate_content(
        contents=[few_shot],
        generation_config=GenerationConfig(temperature=0.0, max_output_tokens=10)
    )
    labels = resp.text.strip().upper().split()
    mapping = []
    if 'STOCK' in labels:       mapping.append('stock')
    if 'MUTUAL_FUND' in labels: mapping.append('mutual_fund')
    if 'LOAN' in labels:        mapping.append('loan')
    return mapping or ['stock']

# --- 4. Analysis Wrappers ---
def analyze_stocks(query: str) -> str:
    prompt = f"""
You are a top-tier stock analyst with access to technical and fundamental data:
{json.dumps(stocks)}

Instructions:
1. If the query asks for a **stock target**, respond with:  
   "Using machine learning technique, the predicted stock target is ₹XYZ".

2. If the query is for **short-term recommendation**, use **technical data** only.

3. If the query is for **long-term investment**, use **both technical and fundamental data**.

4. Answer concisely and justify the choice using the data provided.

Query: {query}
"""
    resp = client.generate_content(contents=[prompt], generation_config=GenerationConfig(temperature=0.5, max_output_tokens=1024))
    return resp.text.strip()

def analyze_mutual_funds(query: str) -> str:
    prompt = f"""
You are an expert financial advisor. You have access to this mutual fund dataset:
{json.dumps(mf_blocks)}

Analyze and answer: {query}
Provide one best suggestion.
"""
    resp = client.generate_content(
        contents=[prompt],
        generation_config=GenerationConfig(temperature=0.5, max_output_tokens=1024)
    )
    return resp.text.strip()

def analyze_loans(query: str) -> str:
    def parse_input(text: str):
        t = text.lower().replace(",", "").replace("₹", "").strip()
        m = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lac|million|crore)?", t)
        amt = int(float(m.group(1)) * {
                    "lakh": 1e5, "lac": 1e5, "million": 1e6, "crore": 1e7
                }.get(m.group(2), 1)) if m else 500000
        y = re.search(r"(\d{1,2})\s*(years?)", text)
        yrs = int(y.group(1)) if y else 5
        return amt, yrs

    def decide_ranking_criteria(ql: str) -> str:
        ql = ql.lower()
        if any(x in ql for x in ["low emi", "small emi"]):
            return "lowest interest rate and longer tenure for lower EMI"
        if any(x in ql for x in ["highest loan", "maximum amount"]):
            return "highest loan amount with reasonable eligibility"
        if any(x in ql for x in ["instant approval", "quick loan"]):
            return "lowest processing fees and fast approval"
        if any(x in ql for x in ["easy eligibility", "low credit score"]):
            return "lowest credit score requirement"
        return "balanced criteria across rate, amount, and eligibility"

    loan_amt, tenure_years = parse_input(query)
    context = "\n\n".join(loan_docs)
    prompt = f"""
You are an expert financial advisor.

User Request:
{query}

Loan Dataset:
{context}

💬 Smart Ranked Response (up to {tenure_years} years):
* **Option 1** …
* **Option 2** …
* **Option 3** …

(Details based on rates, fees, eligibility)
"""
    resp = client.generate_content(
        contents=[prompt],
        generation_config=GenerationConfig(temperature=0.5, max_output_tokens=1024)
    )
    return resp.text.strip()


# --- Add this to chatbot_logic.py ---
        
# --- 5. Router & Runtime ---
def route_query(query: str, user_id: str) -> str:
    # fetch + risk
    user_data = fetch_user_data(user_id,"")
    profile = user_data['profile']
    assets = user_data['assets']
    goal = user_data['financialGoal']
    risk_label = calculate_rule_based_risk(profile, assets)

    # build context
    context = []
    context.append(f"User {profile.get('name')} ({profile.get('age')}y), {profile.get('occupation')}, {profile.get('maritalStatus')}")
    if profile.get('dependents'):
        deps_list = parse_json_field(profile.get('dependents', []))
        deps = [f"{d.get('relationship', 'Unknown')}({d.get('age', '?')}y)" for d in deps_list if isinstance(d, dict)]
    context.append(f"Income: ₹{assets.get('monthlyIncome')}/mo, Exp: ₹{assets.get('monthlyExpenditure')}/mo")
    context.append(f"Savings: ₹{assets.get('currentSavings')}, Emergency Fund: ₹{assets.get('emergencyFund')}")
    context.append(f"Inv: {len(assets.get('investments', []))}, Ins: {len(assets.get('insurance', []))}, Loans: {len(assets.get('loans', []))}")
    context.append(f"Goal: {goal.get('goal')} ₹{goal.get('targetAmount')} by {goal.get('deadline')}")
    context.append(f"Risk Level: {risk_label}")
    prefix = "\n".join(context)
    print(prefix)

    # classify & analyze
    domains = classify_query(query)
    futures = {}
    with ThreadPoolExecutor() as ex:
        if 'stock' in domains:
            futures['Stock'] = ex.submit(analyze_stocks, f"{prefix}\n\n{query}")
        if 'mutual_fund' in domains:
            futures['Mutual Fund'] = ex.submit(analyze_mutual_funds, f"{prefix}\n\n{query}")
        if 'loan' in domains:
            futures['Loan'] = ex.submit(analyze_loans, f"{prefix}\n\n{query}")

    parts = []
    for domain, fut in futures.items():
        parts.append(f"## {domain}\n{fut.result()}")
    return "\n\n".join(parts)
# If run as a script:
if __name__ == "__main__":
    q = input("Enter your financial query:\n")
    print("\n🔄 Processing...\n")
    print(route_query(q))
