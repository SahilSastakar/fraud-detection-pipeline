from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=()
    )

    #AWS creds
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "ap-southeast-2"

    #S3 settings
    s3_bucket_name: str
    s3_raw_prefix: str = "raw/"
    s3_processed_prefix: str = "processed/"
    s3_flagged_prefix: str = "flagged/"

    #pipeline settings
    environment: str = "development"
    log_level: str = "INFO"

    #data generation settings
    num_transactions: int = 10000
    num_customers: int = 500
    fraud_rate: float = 0.02

    #fraud detection thresholds
    velocity_threshold: int = 5
    velocity_window_minutes: int = 10
    amount_zscore_threshold: float = 3.0
    high_risk_amount: float = 5000.0

@lru_cache()
def get_settings():
    return Settings() 