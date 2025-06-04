
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
print(API_BASE_URL)
def fetch_user_data(user_id: str, base_url: str) -> dict:
    """
    Fetch profile, assets, and financialGoal from REST API.
    """
    endpoints = {
        'profile': f"{API_BASE_URL}/api/profiles/{user_id}",
        'assets': f"{API_BASE_URL}/api/assets/{user_id}",
        'financialGoal': f"{API_BASE_URL}/api/financialgoals/{user_id}",
    }
    data = {}
    for key, url in endpoints.items():
        resp = requests.get(url)
        resp.raise_for_status()
        data[key] = resp.json()
    return data


def parse_json_field(field, default=[]):
    """
    Handles both raw lists and JSON stringified lists.
    """
    if isinstance(field, str):
        try:
            parsed = json.loads(field)
            return parsed if isinstance(parsed, list) else default
        except json.JSONDecodeError:
            return default
    elif isinstance(field, list):
        return field
    return default

def calculate_rule_based_risk(profile: dict, assets: dict) -> str:
    """
    Calculate risk category based on user profile and assets.
    Returns: 'Low', 'Medium', or 'High'
    """
    score = 0

    # --- Age ---
    age = profile.get('age', 0)
    if age <= 25:
        score += 10
    elif age <= 40:
        score += 5
    elif age <= 60:
        score += 0
    else:
        score += 5

    # --- Debt-to-Income Ratio ---
    income = float(assets.get('monthlyIncome', 1))
    loans = parse_json_field(assets.get('loans', []))
    total_loan_amt = sum(float(l.get('amount', 0)) for l in loans if isinstance(l, dict))
    dti = (total_loan_amt / (income * 12)) * 100 if income else 0
    if dti < 20:
        score += 0
    elif dti < 40:
        score += 10
    elif dti < 60:
        score += 20
    else:
        score += 30

    # --- Dependents ---
    dep_count = len(profile.get('dependents', []) or [])
    if dep_count == 0:
        score += 0
    elif dep_count == 1:
        score += 5
    else:
        score += 10

    # --- Emergency Fund Coverage ---
    exp = float(assets.get('monthlyExpenditure', 1))
    efund = float(assets.get('emergencyFund', 0))
    months_cov = efund / exp if exp else 0
    if months_cov >= 6:
        score += 0
    elif months_cov >= 3:
        score += 5
    else:
        score += 15

    # --- Investment Buffer ---
    invs = parse_json_field(assets.get('investments', []))
    total_inv = sum(float(i.get('amount', 0)) for i in invs if isinstance(i, dict))
    if total_inv > 2 * income:
        score += 0
    elif total_inv > income:
        score += 5
    else:
        score += 10

    # --- Insurance Coverage ---
    ins_list = parse_json_field(assets.get('insurance', []))
    ins_score = 0
    for ins in ins_list:
        if isinstance(ins, dict):
            cov = float(ins.get('coverage', 0))
            prem = float(ins.get('premium', 1)) * 12
            ratio = cov / prem if prem else 0
            if ratio >= 10:
                ins_score += 0
            elif ratio >= 5:
                ins_score += 5
            else:
                ins_score += 10
    if ins_list:
        score += ins_score / len(ins_list)

    # --- Loan-to-Income Ratio ---
    lti = total_loan_amt / (income * 12) if income else 0
    if lti < 5:
        score += 0
    elif lti < 10:
        score += 10
    else:
        score += 20

    # --- Final Categorization ---
    if score <= 30:
        return 'Low'
    elif score <= 60:
        return 'Medium'
    return 'High'