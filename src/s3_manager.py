# src/s3_manager.py

"""
What this file does: Handles all interactions with AWS S3.
 Creates our data lake bucket, uploads transaction data, 
 downloads it, and lists files. 
This is the first file where we actually talk to AWS using boto3.
"""

import boto3
import io
import json
from typing import Optional
import pandas as pd
from loguru import logger
from botocore.exceptions import ClientError
from mypy_boto3_s3.client import S3Client
from src.config import get_settings

class S3Manager:
    def __init__(self):
        self.settings = get_settings()
        self.client: S3Client = boto3.client(
            "s3",
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key,
            region_name=self.settings.aws_region
        )
        self.bucket = self.settings.s3_bucket_name
        logger.info(f"S3Manager initialised for bucket: {self.bucket}")

    def create_bucket(self) -> bool:
        try:
            self.client.create_bucket(
                Bucket = self.bucket,
                CreateBucketConfiguration = {
                    "LocationConstraint": self.settings.aws_region
                }
            )
            logger.info(f"Created S3 bucket: {self.bucket}")

            self.client.put_bucket_versioning(
                Bucket = self.bucket,
                VersioningConfiguration = {"Status": "Enabled"}
            )
            logger.info(f"Enabled versioning on bucket: {self.bucket}")

            self.client.put_public_access_block(
                Bucket = self.bucket,
                PublicAccessBlockConfiguration = {
                    "BlockPublicAcls": True,
                    "IgnorePublicAcls": True,
                    "BlockPublicPolicy": True,
                    "RestrictPublicBuckets": True
                }
            )
            logger.info(f"Blocked all public access on bucket: {self.bucket}")
            return True
        
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "BucketAlreadyOwnedByYou":
                logger.info(f"Bucket {self.bucket} already exists and is owned by you")
                return True
            else:
                logger.error(f"Failed to create bucket: {e}")
                return False

