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
    """Consume loan applications and score them in real-time"""
    consumer = create_consumer()

    logger.info(f"Starting consumer — listening on topic: {TOPIC}")
    logger.info("Press Ctrl+C to stop")

    stats = {"total": 0, "approve": 0, "review": 0, "decline": 0, "fraud": 0}

    try:
        for message in consumer:
            application = message.value
            app_id = application.get("app_id", "UNKNOWN")

            logger.info(f"Received: {app_id} — Grade: {application['grade']}")

            # Score the application
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

                logger.info(
                    f"✅ {app_id} → "
                    f"Credit: {result.get('credit_risk_score', 'N/A'):.4f} | "
                    f"Fraud: {result.get('fraud_score', 'N/A'):.4f} | "
                    f"Decision: {decision} | "
                    f"Fraud Flag: {'⚠️' if fraud_flag else '✅'}"
                )

                # Print running stats every 5 applications
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