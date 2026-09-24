# Fraud Detection Pipeline

A synthetic-data fraud detection pipeline for AWS: generates realistic transaction data with
injected fraud patterns, validates it, engineers fraud-signal features, and scores transactions
with a rule-based detector. Built on S3 (and Athena) for storage/querying.

## Pipeline stages

1. **Data generation** (`src/data_generator.py`): synthetic transactions with injected fraud
   patterns (e.g. velocity fraud).
2. **Validation** (`src/validator.py`): schema, null, range, category, uniqueness, and format
   checks before data enters the pipeline.
3. **Feature engineering** (`src/feature_engineer.py`): time, velocity, amount, geographic, and
   category-based fraud signals.
4. **Fraud detection** (`src/fraud_detector.py`): weighted rule-based scoring with risk levels
   and precision/recall evaluation against known fraud labels.
5. **Storage** (`src/s3_manager.py`, `src/athena_manager.py`): S3 data lake with versioning and
   public access blocked; Athena for querying.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in your AWS credentials and S3 bucket
```

Configuration is managed via `src/config.py` (pydantic-settings), reading from `.env`.

## Status

Work in progress: `main.py`, `src/pipeline.py`, `src/athena_manager.py`, and the test suite are
not yet implemented.
