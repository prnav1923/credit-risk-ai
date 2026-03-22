import sys
sys.path.insert(0, '/Users/pranav/Code/credit-risk-ai')
from src.database import SessionLocal, Prediction

import json
import logging
import os
import requests
from kafka import KafkaConsumer
from dotenv import load_dotenv  

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "loan-applications"
GROUP_ID = "credit-risk-consumer"
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "credit-risk-secret-key-2024")

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

def create_consumer():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=GROUP_ID,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="earliest",
        enable_auto_commit=True
    )
    logger.info("Kafka consumer created ✅")
    return consumer


def assess_application(application: dict) -> dict:
    """Call FastAPI /v1/assess endpoint"""
    try:
        # Remove app_id before sending to API
        app_data = {k: v for k, v in application.items() if k != "app_id"}

        response = requests.post(
            f"{API_BASE_URL}/v1/assess",
            headers=HEADERS,
            json=app_data,
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def consume_and_score():
    consumer = create_consumer()
    logger.info(f"Starting consumer — listening on topic: {TOPIC}")
    logger.info("Press Ctrl+C to stop")

    stats = {"total": 0, "approve": 0, "review": 0, "decline": 0, "fraud": 0}

    try:
        for message in consumer:
            application = message.value
            app_id = application.get("app_id", "UNKNOWN")

            logger.info(f"Received: {app_id} — Grade: {application['grade']}")

            result = assess_application(application)

            if "error" not in result:
                stats["total"] += 1
                decision = result.get("combined_decision", "UNKNOWN")
                fraud_flag = result.get("fraud_flag", False)

                if decision == "APPROVE":
                    stats["approve"] += 1
                elif decision == "REVIEW":
                    stats["review"] += 1
                elif decision == "DECLINE":
                    stats["decline"] += 1
                if fraud_flag:
                    stats["fraud"] += 1

                # Log to PostgreSQL
                try:
                    db = SessionLocal()
                    app_data = {k: v for k, v in application.items() if k != "app_id"}
                    prediction = Prediction(
                        loan_amnt=application.get("loan_amnt"),
                        int_rate=application.get("int_rate"),
                        annual_inc=application.get("annual_inc"),
                        dti=application.get("dti"),
                        fico_range_low=application.get("fico_range_low"),
                        fico_range_high=application.get("fico_range_high"),
                        grade=application.get("grade"),
                        sub_grade=application.get("sub_grade"),
                        home_ownership=application.get("home_ownership"),
                        purpose=application.get("purpose"),
                        credit_risk_score=result.get("credit_risk_score"),
                        credit_decision=result.get("credit_decision"),
                        credit_risk_level=result.get("credit_risk_level"),
                        fraud_score=result.get("fraud_score"),
                        fraud_flag=result.get("fraud_flag"),
                        fraud_risk=result.get("fraud_risk"),
                        fraud_indicators=result.get("fraud_indicators", []),
                        combined_decision=decision,
                        combined_risk=result.get("combined_risk"),
                        application_json=app_data
                    )
                    db.add(prediction)
                    db.commit()
                    db.close()
                except Exception as db_err:
                    logger.warning(f"PostgreSQL logging failed: {db_err}")

                logger.info(
                    f"✅ {app_id} → "
                    f"Credit: {result.get('credit_risk_score', 'N/A'):.4f} | "
                    f"Fraud: {result.get('fraud_score', 'N/A'):.4f} | "
                    f"Decision: {decision} | "
                    f"Fraud Flag: {'⚠️' if fraud_flag else '✅'}"
                )

                if stats["total"] % 5 == 0:
                    logger.info(
                        f"📊 Stats — Total: {stats['total']} | "
                        f"Approve: {stats['approve']} | "
                        f"Review: {stats['review']} | "
                        f"Decline: {stats['decline']} | "
                        f"Fraud: {stats['fraud']}"
                    )
            else:
                logger.error(f"❌ {app_id} → Error: {result['error']}")

    except KeyboardInterrupt:
        logger.info("\n🛑 Consumer stopped")
        logger.info(f"Final Stats: {stats}")
    finally:
        consumer.close()


if __name__ == "__main__":
    consume_and_score()