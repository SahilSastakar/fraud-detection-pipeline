"""
What this file does: Generates realistic synthetic financial transaction data 
with fraud patterns deliberately injected. This is our raw data source
it mimics what a real bank's transaction feed looks like before it hits the pipeline.
"""

# src/data_generator.py

import uuid
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np
from faker import Faker
from loguru import logger
from src.config import get_settings

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "petrol", "pharmacy",
    "clothing", "electronics", "entertainment", "travel",
    "utilities", "online_retail"
]

MERCHANT_CITIES = [
    "Melbourne", "Sydney", "Brisbane", "Perth",
    "Adelaide", "Canberra", "Gold Coast", "Newcastle"
]

CATEGORY_SPEND_PROFILES = {
    "grocery": (85, 40),
    "restaurant": (45, 25),
    "petrol": (70, 20),
    "pharmacy": (35, 20),
    "clothing": (120, 80),
    "electronics": (350, 200),
    "entertainment": (60, 30),
    "travel": (450, 300),
    "utilities": (150, 50),
    "online_retail": (95, 60),
}

@dataclass
class Transaction:
    transaction_id: str
    customer_id: str
    customer_name: str
    merchant_name: str
    merchant_category: str
    merchant_city: str
    amount: float
    timestamp: datetime
    card_last_four: str
    is_fraud: bool = False
    fraud_type: Optional[str] = None

#custom profile generator

class TransactionGenerator:
    def __init__(self):
        self.settings = get_settings()
        self.fake = Faker('en_AU')
        self.fake.seed_instance(42)
        random.seed(42)
        np.random.seed(42)

    def _generate_customers(self) -> List[dict]:
        logger.info(f"Generating {self.settings.num_customers} customer profiles...")
        customers = []

        for _ in range(self.settings.num_customers):
            home_city = random.choice(MERCHANT_CITIES)
            typical_category = random.choice(MERCHANT_CATEGORIES)
            typical_amount = CATEGORY_SPEND_PROFILES[typical_category][0]

            customers.append({
                "customer_id": str(uuid.uuid4()),
                "customer_name": self.fake.name(),
                "card_last_four": self.fake.numerify("####"),
                "home_city": home_city,
                "typical_spend": round(abs(np.random.normal(typical_amount , typical_amount * 0.3)),2),
                "typical_category": typical_category
                
            })
        logger.info(f"Generated {len(customers)} customer profile")
        return customers

    def _generate_nomrmal_transactions(self, customer: dict, timestamp: datetime) -> Transaction:

        category = random.choice(MERCHANT_CATEGORIES)
        mean_amount , std_amount = CATEGORY_SPEND_PROFILES(category)
        amount = round(abs(np.random.normal(mean_amount , std_amount)),2)

        city_roll = random.random()

        if city_roll < 0.7:
            city = customer["home_city"]
        elif city_roll < 0.9:
            nearby = [c for c in MERCHANT_CITIES if c != customer["home_city"]]
            city = random.random(nearby)
        else:
            city = random.choice(MERCHANT_CITIES)

        return Transaction(
            transaction_id=str(uuid.uuid4()),
            customer_id=customer["customer_id"],
            customer_name=customer["customer_name"],
            merchant_name=self.fake.company(),
            merchant_category=category,
            merchant_city=city,
            amount=amount,
            timestamp=timestamp,
            card_last_four=customer["card_last_four"],
            is_fraud=False,
            fraud_type=None
        )

    def _generate_velocity_fraud(self , customer: dict, base_timestamp: datetime) -> List[Transaction]:

        transactions = []
        num_fraudulent = random.randint(8,15)

        for i in range(num_fraudulent):
            timestamp = base_timestamp + timedelta(minutes=random.randint(0,9))
            amount = round(abs(np.random.normal(200,50)),2)

            transactions.append(Transaction(
                transaction_id=str(uuid.uuid4()),
                customer_id=customer["customer_id"],
                customer_name=customer["customer_name"],
                merchant_name=self.fake.company(),
                merchant_category="online_retail",
                merchant_city=random.choice(MERCHANT_CITIES),
                amount=amount,
                timestamp=timestamp,
                card_last_four=customer["card_last_four"],
                is_fraud=True,
                fraud_type="velocity"
            ))
        logger.debug(f"Generated {num_fraudulent} velocity fraud transactions for customer {customer['customer_id'][:8]}")
        return transactions

    def _generate_amount_fraud(self, customer: dict, timestamp: datetime) -> Transaction:
        fraudulent_amount = round(
            customer["typical_spend"] * random.uniform(8,15),2
        )
        return Transaction(
            transaction_id=str(uuid.uuid4()),
            customer_id=customer["customer_id"],
            customer_name=customer["customer_name"],
            merchant_name=self.fake.company(),
            merchant_category="electronics",
            merchant_city=customer["home_city"],
            amount=fraudulent_amount,
            timestamp=timestamp,
            card_last_four=customer["card_last_four"],
            is_fraud=True,
            fraud_type="amount_anomaly"
        )
    def _generate_geographic_fraud(self, customer: dict, timestamp: datetime) -> Transaction:

        foreign_cities = [c for c in MERCHANT_CITIES if c != customer["home_city"]]
        fraud_city = random.choice(foreign_cities)
        amount = round(abs(np.random.normal(300,100)),2)

        return Transaction(
            transaction_id=str(uuid.uuid4()),
            customer_id=customer["customer_id"],
            customer_name=customer["customer_name"],
            merchant_name=self.fake.company(),
            merchant_category="travel",
            merchant_city=fraud_city,
            amount=amount,
            timestamp=timestamp,
            card_last_four=customer["card_last_four"],
            is_fraud=True,
            fraud_type="geographic"
        )

    def generate(self) -> pd.DataFrame:
        logger.info(f"Starting generation of {self.settings.num_transactions} transactions")
        logger.info(f"Fraud rate: {self.settings.fraud_rate * 100}%")

        customers = self._generate_customers()
        all_transactions: List[Transaction] = []

        start_date = datetime.now() - timedelta(days=90)
        num_fraud_events = int(self.settings.num_transactions * self.settings.fraud_rate)

        fraud_customers = random.sample(customers, min(num_fraud_events, len(customers)))
        fraud_customer_ids = {c["customer_id"] for c in fraud_customers}

        logger.info(f"Injecting fraud patterns into {len(fraud_customers)} customers")

        transactions_per_customer = self.settings.num_transactions // self.settings.num_customers

        for customer in customers:
            for _ in range(transactions_per_customer):
                days_offset = random.randint(0, 89)
                hour = random.randint(6, 23)
                minute = random.randint(0, 59)
                timestamp = start_date + timedelta(
                    days=days_offset,
                    hours=hour,
                    minutes=minute
                )
                transaction = self._generate_normal_transaction(customer, timestamp)
                all_transactions.append(transaction)

            if customer["customer_id"] in fraud_customer_ids:
                fraud_type = random.choice(["velocity", "amount", "geographic"])
                fraud_timestamp = start_date + timedelta(
                    days=random.randint(0, 89),
                    hours=random.randint(6, 23)
                )

                if fraud_type == "velocity":
                    fraud_txns = self._generate_velocity_fraud(customer, fraud_timestamp)
                    all_transactions.extend(fraud_txns)
                elif fraud_type == "amount":
                    all_transactions.append(self._generate_amount_fraud(customer, fraud_timestamp))
                else:
                    all_transactions.append(self._generate_geographic_fraud(customer, fraud_timestamp))

        logger.info(f"Generated {len(all_transactions)} total transactions")

        df = pd.DataFrame([vars(t) for t in all_transactions])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        fraud_count = df["is_fraud"].sum()
        logger.info(f"Total fraudulent transactions: {fraud_count} ({fraud_count/len(df)*100:.2f}%)")

        return df






