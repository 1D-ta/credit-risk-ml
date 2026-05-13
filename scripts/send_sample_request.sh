#!/usr/bin/env bash
set -euo pipefail

API_URL="${1:-http://localhost:8000/predict}"

# Use TMPDIR if set, otherwise fall back to /tmp (works on Unix-like systems)
TMPDIR="${TMPDIR:-/tmp}"
PAYLOAD_FILE="${TMPDIR}/credit_risk_sample_payload.json"
RESPONSE_FILE="${TMPDIR}/credit_risk_sample_response.json"

cat >"${PAYLOAD_FILE}" <<'JSON'
{
  "checking_account_status": "A11",
  "duration_months": 12,
  "credit_history": "A34",
  "purpose": "A40",
  "credit_amount": 5000,
  "savings_account": "A61",
  "employment_status": "A75",
  "installment_rate": 4,
  "personal_status_sex": "A93",
  "other_debtors": "A101",
  "residence_since": 4,
  "property": "A121",
  "age": 67,
  "other_installment_plans": "A143",
  "housing": "A152",
  "existing_credits": 2,
  "job": "A173",
  "num_liable": 1,
  "telephone": "A192",
  "foreign_worker": "A201"
}
JSON

HTTP_CODE=$(curl -s -o "${RESPONSE_FILE}" -w "%{http_code}" \
  -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  --data-binary @"${PAYLOAD_FILE}")

echo "HTTP $HTTP_CODE"
cat "${RESPONSE_FILE}"
echo

# Clean up temporary files
rm -f "${PAYLOAD_FILE}" "${RESPONSE_FILE}"
