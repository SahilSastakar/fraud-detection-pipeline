"""
What this file does: Validates incoming transaction data against a
set of rules 
before it enters the pipeline. If data doesn't meet our standards 
 wrong types, impossible values, missing fields
we catch it here before it corrupts downstream results.
"""

# src/validator.py

# src/validator.py

import pandas as pd
from loguru import logger
from src.config import get_settings

VALID_MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "petrol", "pharmacy",
    "clothing", "electronics", "entertainment", "travel",
    "utilities", "online_retail"
]

VALID_MERCHANT_CITIES = [
    "Melbourne", "Sydney", "Brisbane", "Perth",
    "Adelaide", "Canberra", "Gold Coast", "Newcastle"
]

REQUIRED_COLUMNS = [
    "transaction_id", "customer_id", "customer_name",
    "merchant_name", "merchant_category", "merchant_city",
    "amount", "timestamp", "card_last_four",
    "is_fraud", "fraud_type"
]


class TransactionValidator:

    def __init__(self):
        self.settings = get_settings()

    def _check_schema(self, df: pd.DataFrame) -> list[dict]:
        failures = []
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            failures.append({
                "expectation": "expect_table_columns",
                "column": "table-level",
                "details": f"Missing columns: {missing}"
            })
        return failures

    def _check_nulls(self, df: pd.DataFrame) -> list[dict]:
        failures = []
        null_checked = [
            "transaction_id", "customer_id", "amount",
            "merchant_category", "merchant_city", "timestamp"
        ]
        for col in null_checked:
            if col not in df.columns:
                continue
            null_count = df[col].isnull().sum()
            if null_count > 0:
                failures.append({
                    "expectation": "expect_column_values_to_not_be_null",
                    "column": col,
                    "details": f"{null_count} null values found"
                })
        return failures

    def _check_amounts(self, df: pd.DataFrame) -> list[dict]:
        failures = []
        if "amount" not in df.columns:
            return failures

        invalid = df[(df["amount"] <= 0) | (df["amount"] > 100000)]
        if len(invalid) > 0:
            failures.append({
                "expectation": "expect_column_values_to_be_between",
                "column": "amount",
                "details": f"{len(invalid)} values outside range (0.01, 100000)"
            })
        return failures

    def _check_categories(self, df: pd.DataFrame) -> list[dict]:
        failures = []

        if "merchant_category" in df.columns:
            invalid = df[~df["merchant_category"].isin(VALID_MERCHANT_CATEGORIES)]
            if len(invalid) > 0:
                failures.append({
                    "expectation": "expect_column_values_to_be_in_set",
                    "column": "merchant_category",
                    "details": f"{len(invalid)} invalid categories: {invalid['merchant_category'].unique().tolist()}"
                })

        if "merchant_city" in df.columns:
            invalid = df[~df["merchant_city"].isin(VALID_MERCHANT_CITIES)]
            if len(invalid) > 0:
                failures.append({
                    "expectation": "expect_column_values_to_be_in_set",
                    "column": "merchant_city",
                    "details": f"{len(invalid)} invalid cities"
                })

        return failures

    def _check_uniqueness(self, df: pd.DataFrame) -> list[dict]:
        failures = []
        if "transaction_id" not in df.columns:
            return failures

        duplicate_count = df["transaction_id"].duplicated().sum()
        if duplicate_count > 0:
            failures.append({
                "expectation": "expect_column_values_to_be_unique",
                "column": "transaction_id",
                "details": f"{duplicate_count} duplicate transaction IDs found"
            })
        return failures

    def _check_card_format(self, df: pd.DataFrame) -> list[dict]:
        failures = []
        if "card_last_four" not in df.columns:
            return failures

        invalid = df[~df["card_last_four"].astype(str).str.match(r"^\d{4}$")]
        if len(invalid) > 0:
            failures.append({
                "expectation": "expect_column_values_to_match_regex",
                "column": "card_last_four",
                "details": f"{len(invalid)} values not matching 4-digit format"
            })
        return failures

    def _check_row_count(self, df: pd.DataFrame) -> list[dict]:
        failures = []
        if len(df) < 1 or len(df) > 500000:
            failures.append({
                "expectation": "expect_table_row_count_to_be_between",
                "column": "table-level",
                "details": f"Row count {len(df)} outside allowed range (1, 500000)"
            })
        return failures

    def validate(self, df: pd.DataFrame) -> tuple[bool, dict]:
        logger.info(f"Starting validation of {len(df):,} transactions")

        all_failures = []
        all_failures.extend(self._check_schema(df))
        all_failures.extend(self._check_nulls(df))
        all_failures.extend(self._check_amounts(df))
        all_failures.extend(self._check_categories(df))
        all_failures.extend(self._check_uniqueness(df))
        all_failures.extend(self._check_card_format(df))
        all_failures.extend(self._check_row_count(df))

        total_checks = 7
        failed = len(all_failures)
        passed = total_checks - failed
        success = failed == 0

        validation_report = {
            "total_rows": len(df),
            "total_expectations": total_checks,
            "passed": passed,
            "failed": failed,
            "success": success,
            "failure_details": all_failures
        }

        if success:
            logger.info(f"Validation passed: {passed}/{total_checks} checks met")
        else:
            logger.warning(f"Validation failed: {failed}/{total_checks} checks not met")
            for failure in all_failures:
                logger.warning(f"FAILED: {failure['expectation']} on {failure['column']} — {failure['details']}")

        return success, validation_report