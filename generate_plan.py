import os
import json
import math
from datetime import datetime
from fetch_risk_utils import fetch_user_data, calculate_rule_based_risk, parse_json_field
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# === Gemini API Configuration ===
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
genai.configure(api_key=api_key)
client = genai.GenerativeModel(model_name="gemini-1.5-flash")


# === Load Dataset ===
def load_data(stock_path='nifty100_seqdata.txt',
              mf_path='mutual_fund.txt',
              loan_path='loan_sequential_data.txt'):
    try:
        stocks = [json.loads(line) for line in open(stock_path, 'r')]
        mf_blocks = open(mf_path, 'r', encoding='utf-8').read().strip().split("\n\n")
        loan_docs = open(loan_path, 'r', encoding='utf-8').read().strip().split("\n\n")
    except Exception as e:
        raise FileNotFoundError(f"Error loading datasets: {e}")
    
    print(f"🌟 Loaded {len(stocks)} stocks, {len(mf_blocks)} mutual funds, {len(loan_docs)} loan entries.")
    return stocks, mf_blocks, loan_docs


stocks, mf_blocks, loan_docs = load_data()


# === SIP Calculator ===
def calculate_sip(goal_amount, years_left, annual_return=0.13):
    r = annual_return / 12
    n = years_left * 12
    return goal_amount * r / ((1 + r) ** n - 1)


# === Main Financial Planning Function ===
def generate_financial_plan(user_id: str) -> str:
    user_data = fetch_user_data(user_id, "")
    profile = user_data['profile']
    assets = user_data['assets']
    goal = user_data['financialGoal']
    risk_label = calculate_rule_based_risk(profile, assets)

    # --- Extracted User Info ---
    name = profile.get('name', 'N/A')
    age = profile.get('age', 0)
    profession = profile.get('occupation', 'N/A')
    dependents = profile.get('dependents', [])
    income = assets.get('monthlyIncome', 0)
    expenses = assets.get('monthlyExpenditure', 0)
    savings = assets.get('currentSavings', 0)
    emergency_fund = assets.get('emergencyFund', 0)
    investments = assets.get('investments', [])
    loans = assets.get('loans', [])
    print(savings, emergency_fund, investments, loans)

    emis = sum(l.get('emi', 0) for l in loans)
    surplus = income - expenses - emis

    goal_name = goal.get('goal', 'Unnamed Goal')
    goal_amount = float(goal.get('targetAmount', 0))
    deadline_raw = goal.get('deadline')
    deadline_year = int(deadline_raw.split('-')[0]) if isinstance(deadline_raw, str) else int(deadline_raw)
    years_left = max(1, deadline_year - datetime.now().year)

    sip_required = calculate_sip(goal_amount, years_left)

    # --- Mutual Fund Parsing ---



    # === Prompt to Gemini ===
    prompt = f"""
You are a certified financial planner. Create a professional markdown report that helps the user achieve their goal.

## 🎯 Goal Plan for {goal_name} – ₹{goal_amount:,.0f} by {deadline_year}

### 👤 User Snapshot
- Name: {name} ({age}y, {profession})
- Dependents: {', '.join([d.get('relation', '') + f" ({d.get('age', '')}y)" for d in dependents]) or 'None'}
- Monthly Income: ₹{income:,.0f} → grows 10%/yr
- Monthly Expense: ₹{expenses:,.0f}
- Current Savings: ₹{savings:,.0f} | Emergency Fund: ₹{emergency_fund:,.0f}
- Investments: {len(investments)} items
- Loans: {len(loans)} active (Total EMI: ₹{emis:,.0f}/mo)

---

### 📊 Goal Summary
- Target: ₹{goal_amount:,.0f} in {years_left} years
- Required SIP (assuming 13% return): ₹{math.ceil(sip_required):,.0f}/mo
- Current monthly surplus: ₹{surplus:,.0f}
- Risk Profile: {risk_label}

---

### 💰 Suggested Action Plans

Generate two risk-based investment plans based on `{risk_label}` profile:
1. Moderate Plan
2. Aggressive Plan

Each plan must:
- Allocate SIP into mutual funds, stocks, gold, and fixed deposits based on dataset provided
- Justify choices based on risk and goal horizon
- Suggest SIP step-up aligned with salary growth (10%/yr)
- Include at least 3 top mutual funds and 3 stock picks from below datasets along with analysis
- Suggest term insurance if no life cover (10x income)
- Include loan optimization advice and potential savings
- Show projection table of savings growth
- Recommend how to build up emergency fund to 6x monthly expenses, if short

---

### 📂 Mutual Fund Dataset
{json.dumps(mf_blocks[:40], indent=2)}

---

### 📈 Stock Dataset
{json.dumps(stocks[:40], indent=2)}

---

### 🧾 Final Output Format:
Respond in **Markdown** using these headers:
show  montly amount invesmtent in each type of investment based on calculated requirment sip to acheive goal.
1. 💰 Suggested Action Plan (Moderate)  with Mutual Fund Recommendations  and stock recommendations
2. 💰 Suggested Action Plan (Aggressive)   with Mutual Fund Recommendations  and stock recommendations
3. 📉 Loan Optimization Advice  
4. 📊 Projection Table for both plan for comparison which one is better along with risk   
5. ✅ Final Advice (with confidence)

Avoid bullet spamming. Make it look like a real wealth advisor's report like professional.
"""

    response = client.generate_content(
        [prompt],
        generation_config=GenerationConfig(temperature=0.7, max_output_tokens=2096)
    )

    return response.text.strip()

# Example usage for testing
if __name__ == "__main__":
    print(generate_financial_plan())
