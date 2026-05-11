# Product Requirements Document

## Project
Production-Grade Credit Risk Scoring System

## Goal
Build a local, production-shaped credit risk system that prioritizes schema discipline, deterministic training, approval gating, drift detection, manual rollback, and auditability over raw model performance.

## Non-Goals
- No deep learning or complex ensembles
- No automated retraining or self-healing
- No UI/dashboard
- No authentication, authorization, or PII masking
- No claim of real regulatory compliance

## Data Source
- Dataset: Statlog (German Credit Data)
- Source file: raw space-separated `german.data`
- Size: 1000 rows, 20 input features, 1 target
- Target: `1` = good credit, `2` = bad credit

## Functional Requirements
1. Strict raw-data ingestion with explicit schema validation.
2. Deterministic training with a single auditable model.
3. Post-training calibration and an approval gate before serving.
4. Minimal FastAPI inference service that loads only approved models.
5. PSI/KS drift detection against training distributions.
6. Manual rollback to the last approved model.
7. Local execution via Docker Compose.

## Failure Modes To Demonstrate
- Schema mismatch
- Unexpected categorical value
- Degraded calibration
- Feature drift
- Bad model deployment

## Acceptance Criteria
- The system runs end to end locally.
- At least one failure is injected and explained.
- Rollback is executed and verified.
- The README explains the design choices clearly.