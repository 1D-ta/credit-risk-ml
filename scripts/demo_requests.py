import os
from pathlib import Path

import requests

# Get project root (2 levels up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

URL = os.environ.get("API_URL", "http://localhost:8000/predict")

# build a payload from the first line of the timestamped data
data_file = PROJECT_ROOT / "data" / "raw" / "german_credit_with_ts.txt"
with open(data_file, "r") as f:
    first = f.readline().strip().split()

# schema: timestamp + original tokens
tokens = first[1:]
fields = [
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_status",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_liable",
    "telephone",
    "foreign_worker",
]

numeric_fields = set(["duration_months", "credit_amount", "installment_rate", "residence_since", "age", "existing_credits", "num_liable"])

payload = {}
for k, v in zip(fields, tokens):
    if k in numeric_fields:
        payload[k] = int(v)
    else:
        payload[k] = v

print("Sending valid payload:\n", payload)
resp = requests.post(URL, json=payload, timeout=10)
print("Valid response:", resp.status_code, resp.text)

# shifted payload (credit_amount x100)
bad = dict(payload)
bad["credit_amount"] = bad["credit_amount"] * 100
print("\nSending shifted payload (credit_amount x100):\n", bad)
resp2 = requests.post(URL, json=bad, timeout=10)
print("Shifted response:", resp2.status_code, resp2.text)
