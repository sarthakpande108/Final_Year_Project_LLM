# generate_plan.py

# chatbot_logic.py

import os
import json
import math
from datetime import datetime
from fetch_risk_utils import fetch_user_data, calculate_rule_based_risk, parse_json_field
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Gemini API setup
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
client = genai.GenerativeModel(model_name="gemini-1.5-flash")
# Replace with your actual API base URL
# Load datasets
def load_data(stock_path='nifty100_seqdata.txt',
              mf_path='mutual_fund_sequential_data.txt',
              loan_path='loan_sequential_data.txt'):
    stocks = [json.loads(line) for line in open(stock_path, 'r')]
    mf_blocks = open(mf_path, 'r', encoding='utf-8').read().strip().split("\n\n")
    loan_docs = open(loan_path, 'r', encoding='utf-8').read().strip().split("\n\n")
    print(f"🌟 Loaded {len(stocks)} stocks, {len(mf_blocks)} mutual funds, {len(loan_docs)} loan entries.")
    return stocks, mf_blocks, loan_docs

stocks, mf_blocks, loan_docs = load_data()

# SIP calculator
def calculate_sip(goal_amount, years_left, annual_return=0.13):
    r = annual_return / 12
    n = years_left * 12
    return goal_amount * r / ((1 + r) ** n - 1)

# Main financial planning function
def generate_financial_plan(user_id: str) -> str:
    user_data = fetch_user_data(user_id, "")
    profile = user_data['profile']
    assets = user_data['assets']
    goal = user_data['financialGoal']
    risk_label = calculate_rule_based_risk(profile, assets)

    income = assets.get('monthlyIncome', 0)
    expenses = assets.get('monthlyExpenditure', 0)
    emis = sum(l.get('emi', 0) for l in assets.get('loans', []))
    surplus = income - expenses - emis

    goal_amount = float(goal.get('targetAmount', 0))
    deadline_raw = goal.get('deadline')
    deadline_year = int(deadline_raw.split('-')[0]) if isinstance(deadline_raw, str) else int(deadline_raw)
    years_left = max(1, deadline_year - datetime.now().year)

    sip_required = calculate_sip(goal_amount, years_left)

    # Parse mutual funds for JSON formatting
    parsed_mfs = []
    for block in mf_blocks:
        try:
            mf = parse_json_field(block)
            parsed_mfs.append(mf)
        except Exception:
            continue

    # Generate Gemini Prompt
    full_prompt = f"""
You are a SEBI-registered financial advisor.

📍 Client Profile:
Name: {profile.get('name')}
Risk Tolerance: {risk_label}
Goal: {goal.get('goal')}
Target Amount: ₹{goal_amount}
Deadline: {years_left} years
Monthly Income: ₹{income}
Expenses: ₹{expenses}
EMIs: ₹{emis}
Monthly Surplus: ₹{surplus}
Required SIP: ₹{math.ceil(sip_required)}

🧾 Mutual Fund Dataset:
{json.dumps(parsed_mfs[:100], indent=2)}

📊 Stock Dataset:
{json.dumps(stocks[:100], indent=2)}

Instructions:
- Suggest top 3 mutual funds (mix of ELSS, debt, hybrid) based on risk, returns, and goal horizon.
- Suggest top 3 stocks aligned with risk profile and goal.
- Recommend allocation across mutual funds, stocks, gold, savings/FD.
- Calculate monthly amounts for each asset class based on SIP needed.
- Show expected returns for each category and final goal projection.
- Output in an intelligent, logical, and professional tone.

Respond in this format:
1. 🎯 Goal Summary + SIP Feasibility
2. 📊 Investment Allocation Strategy
3. 🧠 Mutual Fund Recommendations
4. 📈 Stock Recommendations
5. 🪙 Gold/Savings Plan
6. 🔎 Projected Outcome Table
7. ✅ Goal Status + Final Advice
"""

    response = client.generate_content([full_prompt], generation_config=GenerationConfig(temperature=0.3, max_output_tokens=4096))
    return response.text.strip()

# Example usage for testing
if __name__ == "__main__":
    print(generate_financial_plan())
