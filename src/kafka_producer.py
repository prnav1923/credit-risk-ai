import json
import logging
import time
import random
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "loan-applications"


def create_producer():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8")
    )
    logger.info("Kafka producer created ✅")
    return producer


def generate_loan_application(app_id: int) -> dict:
    """Generate a synthetic loan application"""
    grades = ["A", "B", "C", "D", "E", "F"]
    sub_grades = {
        "A": ["A1", "A2", "A3", "A4", "A5"],
        "B": ["B1", "B2", "B3", "B4", "B5"],
        "C": ["C1", "C2", "C3", "C4", "C5"],
        "D": ["D1", "D2", "D3", "D4", "D5"],
        "E": ["E1", "E2", "E3", "E4", "E5"],
        "F": ["F1", "F2", "F3", "F4", "F5"]
    }
    purposes = [
        "debt_consolidation", "credit_card", "home_improvement",
        "major_purchase", "medical", "car", "other"
    ]
    home_ownership = ["RENT", "OWN", "MORTGAGE"]

    grade = random.choice(grades)
    fico_base = random.randint(580, 820)

    return {
        "app_id": f"APP_{app_id:06d}",
        "loan_amnt": random.choice([5000, 10000, 15000, 20000, 25000, 30000]),
        "int_rate": round(random.uniform(5.5, 30.0), 2),
        "installment": round(random.uniform(100, 900), 2),
        "annual_inc": random.choice([30000, 45000, 60000, 80000, 100000, 150000]),
        "dti": round(random.uniform(5, 45), 1),
        "delinq_2yrs": random.choice([0, 0, 0, 1, 2, 3]),
        "fico_range_low": fico_base,
        "fico_range_high": fico_base + 4,
        "open_acc": random.randint(3, 20),
        "pub_rec": random.choice([0, 0, 0, 0, 1]),
        "revol_bal": random.randint(1000, 50000),
        "revol_util": round(random.uniform(10, 95), 1),
        "total_acc": random.randint(5, 40),
        "emp_length": random.randint(0, 10),
        "mort_acc": random.randint(0, 5),
        "pub_rec_bankruptcies": random.choice([0, 0, 0, 0, 1]),
        "num_actv_bc_tl": random.randint(1, 10),
        "bc_util": round(random.uniform(10, 90), 1),
        "percent_bc_gt_75": round(random.uniform(0, 80), 1),
        "avg_cur_bal": random.randint(1000, 20000),
        "home_ownership": random.choice(home_ownership),
        "verification_status": random.choice(["Verified", "Source Verified", "Not Verified"]),
        "purpose": random.choice(purposes),
        "grade": grade,
        "sub_grade": random.choice(sub_grades[grade]),
        "initial_list_status": random.choice(["w", "f"]),
        "application_type": "Individual"
    }


def produce_applications(count: int = 10, delay: float = 1.0):
    """Produce loan applications to Kafka topic"""
    producer = create_producer()

    logger.info(f"Producing {count} loan applications to topic: {TOPIC}")

    for i in range(1, count + 1):
        application = generate_loan_application(i)
        key = application["app_id"]

        producer.send(
            TOPIC,
            key=key,
            value=application
        )

        logger.info(
            f"Sent APP_{i:06d} — "
            f"Grade: {application['grade']}, "
            f"Amount: ${application['loan_amnt']:,}, "
            f"FICO: {application['fico_range_low']}"
        )

        time.sleep(delay)

    producer.flush()
    producer.close()
    logger.info(f"✅ Produced {count} applications successfully")


if __name__ == "__main__":
    produce_applications(count=10, delay=0.5)