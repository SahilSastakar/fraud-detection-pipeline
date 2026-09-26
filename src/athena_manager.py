# src/athena_manager.py

import boto3
import time
import io
import pandas as pd
from loguru import logger
from botocore.exceptions import ClientError
from src.config import get_settings


class AthenaManager:

    def __init__(self):
        self.settings = get_settings()
        self.client = boto3.client(
            "athena",
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key,
            region_name=self.settings.aws_region
        )
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key,
            region_name=self.settings.aws_region
        )
        self.database = "fraud_intelligence"
        self.results_prefix = "athena-results/"
        self.results_location = f"s3://{self.settings.s3_bucket_name}/{self.results_prefix}"
        logger.info(f"AthenaManager initialised. Results location: {self.results_location}")

    def create_database(self) -> bool:
        logger.info(f"Creating Athena database: {self.database}")
        query = f"CREATE DATABASE IF NOT EXISTS {self.database}"
        success, _ = self._execute_query(query)
        if success:
            logger.info(f"Database {self.database} ready")
        return success

    def create_transactions_table(self) -> bool:
        logger.info("Creating Athena table for processed transactions")

        s3_location = f"s3://{self.settings.s3_bucket_name}/{self.settings.s3_processed_prefix}"

        query = f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS {self.database}.transactions (
            transaction_id STRING,
            customer_id STRING,
            customer_name STRING,
            merchant_name STRING,
            merchant_category STRING,
            merchant_city STRING,
            amount DOUBLE,
            timestamp TIMESTAMP,
            card_last_four STRING,
            is_fraud BOOLEAN,
            fraud_type STRING,
            hour_of_day INT,
            day_of_week INT,
            is_weekend INT,
            is_night INT,
            is_business_hours INT,
            velocity_count DOUBLE,
            is_high_velocity INT,
            customer_mean_amount DOUBLE,
            customer_std_amount DOUBLE,
            customer_max_amount DOUBLE,
            customer_transaction_count BIGINT,
            amount_zscore DOUBLE,
            is_high_amount INT,
            is_amount_anomaly INT,
            home_city STRING,
            is_away_from_home INT,
            time_since_last_txn DOUBLE,
            is_rapid_location_change INT,
            typical_category STRING,
            is_unusual_category INT,
            fraud_score DOUBLE,
            risk_level STRING,
            triggered_rules STRING
        )
        STORED AS PARQUET
        LOCATION '{s3_location}'
        TBLPROPERTIES ('parquet.compression'='SNAPPY')
        """

        success, _ = self._execute_query(query)
        if success:
            logger.info("Transactions table created successfully")
        return success

    def create_alerts_table(self) -> bool:
        logger.info("Creating Athena table for fraud alerts")

        s3_location = f"s3://{self.settings.s3_bucket_name}/{self.settings.s3_flagged_prefix}"

        query = f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS {self.database}.fraud_alerts (
            transaction_id STRING,
            customer_id STRING,
            customer_name STRING,
            amount DOUBLE,
            merchant_category STRING,
            merchant_city STRING,
            timestamp STRING,
            fraud_score DOUBLE,
            risk_level STRING,
            triggered_rules STRING,
            is_known_fraud BOOLEAN
        )
        STORED AS PARQUET
        LOCATION '{s3_location}'
        TBLPROPERTIES ('parquet.compression'='SNAPPY')
        """

        success, _ = self._execute_query(query)
        if success:
            logger.info("Fraud alerts table created successfully")
        return success

    def _execute_query(self, query: str) -> tuple[bool, str]:
        try:
            response = self.client.start_query_execution(
                QueryString=query,
                QueryExecutionContext={"Database": self.database} if self.database not in query else {},
                ResultConfiguration={"OutputLocation": self.results_location}
            )

            query_execution_id = response["QueryExecutionId"]
            logger.debug(f"Query submitted. ExecutionId: {query_execution_id}")

            while True:
                status_response = self.client.get_query_execution(
                    QueryExecutionId=query_execution_id
                )

                state = status_response["QueryExecution"]["Status"]["State"]

                if state == "SUCCEEDED":
                    logger.debug(f"Query {query_execution_id} succeeded")
                    return True, query_execution_id

                elif state in ["FAILED", "CANCELLED"]:
                    reason = status_response["QueryExecution"]["Status"].get(
                        "StateChangeReason", "Unknown reason"
                    )
                    logger.error(f"Query {query_execution_id} {state}: {reason}")
                    return False, query_execution_id

                else:
                    logger.debug(f"Query {query_execution_id} state: {state}. Waiting...")
                    time.sleep(2)

        except ClientError as e:
            logger.error(f"Failed to execute Athena query: {e}")
            return False, ""

    def query_to_dataframe(self, query: str) -> pd.DataFrame | None:
        logger.info(f"Executing Athena query: {query[:100]}...")

        success, execution_id = self._execute_query(query)

        if not success:
            return None

        try:
            results_key = f"{self.results_prefix}{execution_id}.csv"
            response = self.s3_client.get_object(
                Bucket=self.settings.s3_bucket_name,
                Key=results_key
            )

            body = response["Body"].read()
            df = pd.read_csv(io.BytesIO(body))

            logger.info(f"Query returned {len(df):,} rows")
            return df

        except ClientError as e:
            logger.error(f"Failed to retrieve query results: {e}")
            return None

    def run_fraud_summary_queries(self) -> dict:
        logger.info("Running fraud intelligence summary queries")

        queries = {
            "risk_distribution": f"""
                SELECT risk_level, COUNT(*) as count,
                       ROUND(AVG(fraud_score), 4) as avg_score
                FROM {self.database}.fraud_alerts
                GROUP BY risk_level
                ORDER BY avg_score DESC
            """,

            "top_flagged_categories": f"""
                SELECT merchant_category,
                       COUNT(*) as alert_count,
                       ROUND(AVG(fraud_score), 4) as avg_score,
                       ROUND(AVG(amount), 2) as avg_amount
                FROM {self.database}.fraud_alerts
                GROUP BY merchant_category
                ORDER BY alert_count DESC
                LIMIT 5
            """,

            "city_risk_profile": f"""
                SELECT merchant_city,
                       COUNT(*) as alert_count,
                       SUM(CASE WHEN risk_level = 'CRITICAL' THEN 1 ELSE 0 END) as critical_count
                FROM {self.database}.fraud_alerts
                GROUP BY merchant_city
                ORDER BY alert_count DESC
            """,

            "detection_accuracy": f"""
                SELECT
                    COUNT(*) as total_alerts,
                    SUM(CASE WHEN is_known_fraud = true THEN 1 ELSE 0 END) as true_positives,
                    SUM(CASE WHEN is_known_fraud = false THEN 1 ELSE 0 END) as false_positives,
                    ROUND(
                        SUM(CASE WHEN is_known_fraud = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
                        2
                    ) as precision_pct
                FROM {self.database}.fraud_alerts
            """
        }

        results = {}
        for query_name, query in queries.items():
            logger.info(f"Running query: {query_name}")
            df = self.query_to_dataframe(query)
            if df is not None:
                results[query_name] = df
                logger.info(f"Query {query_name} returned {len(df)} rows")
            else:
                logger.warning(f"Query {query_name} failed or returned no results")

        return results