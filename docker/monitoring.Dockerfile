FROM python:3.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD ["/bin/sh", "-c", "python monitoring/drift.py --reference-data data/raw/german_credit_raw.txt --current-data data/raw/german_credit_raw.txt --schema data/schemas/schema.json"]