FROM python:3.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD ["/bin/sh", "-c", "python training/train.py --raw-data data/raw/german_credit_raw.txt --schema data/schemas/schema.json && python training/evaluate.py --raw-data data/raw/german_credit_raw.txt --schema data/schemas/schema.json --model-artifact artifacts/models/model_v1.pkl && python training/calibration.py --raw-data data/raw/german_credit_raw.txt --schema data/schemas/schema.json --model-artifact artifacts/models/model_v1.pkl && python governance/approve_model.py --model-artifact artifacts/models/calibrated_model_v1.pkl --metrics artifacts/reports/metrics.json --calibration-report artifacts/reports/calibration_report.json && python governance/rollback.py"]