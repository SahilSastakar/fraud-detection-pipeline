"""
What this file does: Takes the feature-engineered DataFrame 
and applies a scoring system to flag suspicious transactions.
each transaction gets a fraud score between 0 and 1, a risk level, 
and a list of the specific rules that triggered. 
This is the intelligence layer of the pipeline.
"""
# src/fraud_detector.py

from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np
from loguru import logger
from src.config import get_settings

@dataclass
class FraudAlert:
    transaction_id: str
    customer_id: str
    customer_name: str
    amount: float
    merchant_category: str
    merchant_city: str
    timestamp: str
    fraud_score: float
    risk_level: str
    triggered_rules: List[str]
    is_known_fraud: bool

FRAUD_RULES = {
    "high_velocity": {
        "weight": 0.35,
        "description": "Unusually high number of transactions in a short window"
    },
    "amount_anomaly": {
        "weight": 0.30,
        "description": "Transaction amount significantly above customer baseline"
    },
    "high_absolute_amount": {
        "weight": 0.20,
        "description": "Transaction amount exceeds high-risk threshold"
    },
    "rapid_location_change": {
        "weight": 0.25,
        "description": "Transaction in unusual location shortly after previous transaction"
    },
    "night_transaction": {
        "weight": 0.10,
        "description": "Transaction occurred during overnight hours"
    },
    "unusual_category": {
        "weight": 0.10,
        "description": "Transaction in merchant category unusual for this customer"
    },
    "new_customer_high_spend": {
        "weight": 0.15,
        "description": "Customer with few transactions making a high-value purchase"
    },
}

class FraudDetector:

    def __init__(self):
        self.settings = get_settings()
        self.low_threshold = 0.3
        self.medium_threshold = 0.5
        self.high_threshold = 0.7

    def _get_risk_level(self, score: float) -> str:
        if score >= self.high_threshold:
            return "CRITICAL"
        elif score >= self.medium_threshold:
            return "HIGH"
        elif score >= self.low_threshold:
            return "MEDIUM"
        else:
            return "LOW"

    def _score_transaction(self, row: pd.Series) -> tuple[float, List[str]]:
        triggered_rules = []
        raw_score = 0.0

        if row.get("is_high_velocity", 0) == 1:
            triggered_rules.append("high_velocity")
            raw_score += FRAUD_RULES["high_velocity"]["weight"]

        if row.get("is_amount_anomaly", 0) == 1:
            triggered_rules.append("amount_anomaly")
            raw_score += FRAUD_RULES["amount_anomaly"]["weight"]

        if row.get("is_high_amount", 0) == 1:
            triggered_rules.append("high_absolute_amount")
            raw_score += FRAUD_RULES["high_absolute_amount"]["weight"]

        if row.get("is_rapid_location_change", 0) == 1:
            triggered_rules.append("rapid_location_change")
            raw_score += FRAUD_RULES["rapid_location_change"]["weight"]

        if row.get("is_night", 0) == 1:
            triggered_rules.append("night_transaction")
            raw_score += FRAUD_RULES["night_transaction"]["weight"]

        if row.get("is_unusual_category", 0) == 1:
            triggered_rules.append("unusual_category")
            raw_score += FRAUD_RULES["unusual_category"]["weight"]

        customer_count = row.get("customer_transaction_count", 10)
        if customer_count <= 3 and row.get("is_high_amount", 0) == 1:
            triggered_rules.append("new_customer_high_spend")
            raw_score += FRAUD_RULES["new_customer_high_spend"]["weight"]

        final_score = min(raw_score, 1.0)
        return final_score, triggered_rules

    def detect(self, df: pd.DataFrame) -> tuple[pd.DataFrame, List[FraudAlert]]:
        logger.info(f"Running fraud detection on {len(df):,} transactions")

        fraud_scores = []
        risk_levels = []
        triggered_rules_list = []

        for _, row in df.iterrows():
            score, rules = self._score_transaction(row)
            fraud_scores.append(score)
            risk_levels.append(self._get_risk_level(score))
            triggered_rules_list.append(rules)

        df["fraud_score"] = fraud_scores
        df["risk_level"] = risk_levels
        df["triggered_rules"] = triggered_rules_list

        flagged_df = df[df["risk_level"].isin(["MEDIUM", "HIGH", "CRITICAL"])].copy()

        alerts = []
        for _, row in flagged_df.iterrows():
            alert = FraudAlert(
                transaction_id=row["transaction_id"],
                customer_id=row["customer_id"],
                customer_name=row["customer_name"],
                amount=row["amount"],
                merchant_category=row["merchant_category"],
                merchant_city=row["merchant_city"],
                timestamp=str(row["timestamp"]),
                fraud_score=row["fraud_score"],
                risk_level=row["risk_level"],
                triggered_rules=row["triggered_rules"],
                is_known_fraud=row.get("is_fraud", False)
            )
            alerts.append(alert)

        total_flagged = len(flagged_df)
        critical = len(df[df["risk_level"] == "CRITICAL"])
        high = len(df[df["risk_level"] == "HIGH"])
        medium = len(df[df["risk_level"] == "MEDIUM"])

        logger.info(f"Fraud detection complete:")
        logger.info(f"  Total transactions: {len(df):,}")
        logger.info(f"  Flagged: {total_flagged:,} ({total_flagged/len(df)*100:.2f}%)")
        logger.info(f"  CRITICAL: {critical:,}")
        logger.info(f"  HIGH: {high:,}")
        logger.info(f"  MEDIUM: {medium:,}")

        return df, alerts

    def evaluate(self, df: pd.DataFrame) -> dict:
        logger.info("Evaluating fraud detection performance")

        if "is_fraud" not in df.columns or "risk_level" not in df.columns:
            logger.warning("Cannot evaluate — missing ground truth or predictions")
            return {}

        actual_fraud = df["is_fraud"]
        predicted_fraud = df["risk_level"].isin(["MEDIUM", "HIGH", "CRITICAL"])

        true_positives = ((actual_fraud == True) & (predicted_fraud == True)).sum()
        false_positives = ((actual_fraud == False) & (predicted_fraud == True)).sum()
        true_negatives = ((actual_fraud == False) & (predicted_fraud == False)).sum()
        false_negatives = ((actual_fraud == True) & (predicted_fraud == False)).sum()

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            "true_positives": int(true_positives),
            "false_positives": int(false_positives),
            "true_negatives": int(true_negatives),
            "false_negatives": int(false_negatives),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(false_positives / (false_positives + true_negatives), 4) if (false_positives + true_negatives) > 0 else 0
        }

        logger.info(f"Detection metrics:")
        logger.info(f"  Precision: {metrics['precision']:.2%}")
        logger.info(f"  Recall: {metrics['recall']:.2%}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"  False Positive Rate: {metrics['false_positive_rate']:.2%}")

        return metrics