import os
import logging
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    Boolean, DateTime, JSON, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/credit_risk_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Table 1: Predictions ---
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Input features
    loan_amnt = Column(Float)
    int_rate = Column(Float)
    annual_inc = Column(Float)
    dti = Column(Float)
    fico_range_low = Column(Float)
    fico_range_high = Column(Float)
    grade = Column(String)
    sub_grade = Column(String)
    home_ownership = Column(String)
    purpose = Column(String)

    # Credit risk output
    credit_risk_score = Column(Float)
    credit_decision = Column(String)
    credit_risk_level = Column(String)

    # Fraud output
    fraud_score = Column(Float)
    fraud_flag = Column(Boolean)
    fraud_risk = Column(String)
    fraud_indicators = Column(JSON)

    # Combined
    combined_decision = Column(String)
    combined_risk = Column(String)

    # Full application JSON
    application_json = Column(JSON)


# --- Table 2: Agent Decisions ---
class AgentDecision(Base):
    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    agent_type = Column(String)  # "single" or "multi"
    application_json = Column(JSON)
    report = Column(Text)
    decision = Column(String)


# --- Table 3: Overrides ---
class Override(Base):
    __tablename__ = "overrides"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    reviewer_id = Column(String)
    original_decision = Column(String)
    override_decision = Column(String)
    reason = Column(Text)
    risk_score = Column(Float)
    application_json = Column(JSON)


# --- Table 4: Drift Reports ---
class DriftReport(Base):
    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    status = Column(String)
    baseline_auc = Column(Float)
    current_auc = Column(Float)
    drift = Column(Float)
    drift_pct = Column(Float)
    action = Column(String)


# --- Create all tables ---
def init_db():
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created ✅")


# --- Get DB session ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()