"""
What this file does: Takes the raw validated transaction data and 
adds new columns that capture fraud signals.
 These new columns are called features — derived measurements 
that make patterns visible to the fraud detection logic.
"""
# src/feature_engineer.py

import pandas as pd
import numpy as np
from loguru import logger
from src.config import get_settings

class FeatureEngineer:
    def __init__(self):
        self.settings = get_settings()

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Adding time based features")

        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
        df["is_night"] = ((df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 5)).astype(int)
        df["is_business_hours"] = (
            (df["hour_of_day"] >= 9) &
            (df["hour_of_day"] <= 17) &
            (df["is_weekend"] == 0)
        ).astype(int)

        return df

    def _add_velocity_features(self , df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding velocity features")

        df = df.sort_values("timestamp").reset_index(drop=True)
        window = f"{self.settings.velocity_window_minutes}min"

        velocity_counts = (
            df.groupby("customer_id")
            .apply(
                lambda x: x.set_index("timestamp")["transaction_id"]
                .rolling(window, closed="both")
                .count()
                .reset_index()

            )
            .reset_index(drop=True)
        )

        velocity_counts.columns = ["timestamp", "velocity_count"]
        velocity_counts["customer_id"] = df["customer_id"].values

        df = df.merge(
            velocity_counts[["customer_id", "timestamp", "velocity_count"]],
            on=["customer_id", "timestamp"],
            how="left"
        )

        df["velocity_count"] = df["velocity_count"].fillna(1)
        df["is_high_velocity"] = (
            df["velocity_count"] >= self.settings.velocity_threshold
        ).astype(int)

        return df

    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding amount-based features")

        customer_stats = (
            df.groupby("customer_id")["amount"]
            .agg(
                customer_mean_amount="mean",
                customer_std_amount="std",
                customer_max_amount="max",
                customer_transaction_count="count"
            )
            .reset_index()
        )
        customer_stats["customer_std_amount"] = customer_stats["customer_std_amount"].fillna(1)

        df = df.merge(customer_stats, on= "customer_id", how="left")

        df["amount_zscore"] = ((df["amount"] - df["customer_mean_amount"]) / df["customer_std_amount"])

        df["amount_zscore"] = df["amount_zscore"].fillna(0)

        df["is_high_amount"] = (
            df["amount"] >= self.settings.high_risk_amount
        ).astype(int)

        df["is_amount_anomaly"] = (
            df["amount_zscore"] >= self.settings.amount_zscore_threshold
        ).astype(int)

        return df

    def _add_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding geographic features")

        customer_home_city = (
            df.groupby("customer_id")["merchant_city"]
            .agg(lambda x: x.mode()[0])
            .reset_index()
            .rename(columns={"merchant_city": "home_city"})
        )

        df = df.merge(customer_home_city, on="customer_id", how="left")

        df["is_away_from_home"] = (
            df["merchant_city"] != df["home_city"]
        ).astype(int)

        df["time_since_last_txn"] = (
            df.groupby("customer_id")["timestamp"]
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )

        df["is_rapid_location_change"] = (
            (df["is_away_from_home"] == 1) &
            (df["time_since_last_txn"] < 3600)
        ).astype(int)

        return df

    def _add_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding category-based features")

        customer_category = (
            df.groupby("customer_id")["merchant_category"]
            .agg(lambda x: x.mode()[0])
            .reset_index()
            .rename(columns={"merchant_category": "typical_category"})
        )

        df = df.merge(customer_category, on="customer_id", how="left")

        df["is_unusual_category"] = (
            df["merchant_category"] != df["typical_category"]
        ).astype(int)

        return df

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting feature engineering on {len(df):,} transactions")

        df = self._add_time_features(df)
        df = self._add_velocity_features(df)
        df = self._add_amount_features(df)
        df = self._add_geographic_features(df)
        df = self._add_category_features(df)

        feature_columns = [
            "hour_of_day", "day_of_week", "is_weekend", "is_night",
            "is_business_hours", "velocity_count", "is_high_velocity",
            "customer_mean_amount", "customer_std_amount", "amount_zscore",
            "is_high_amount", "is_amount_anomaly", "home_city",
            "is_away_from_home", "time_since_last_txn",
            "is_rapid_location_change", "typical_category",
            "is_unusual_category"
        ]

        logger.info(f"Feature engineering complete. Added {len(feature_columns)} features")
        logger.info(f"Final dataset shape: {df.shape}")

        return df
