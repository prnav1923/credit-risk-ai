# 🏦 Credit Risk Assessment AI System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green?logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)
![LangChain](https://img.shields.io/badge/LangChain-1.2-purple)
![AWS](https://img.shields.io/badge/AWS-ECS%20Fargate-FF9900?logo=amazonaws)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions)
![MLflow](https://img.shields.io/badge/MLflow-3.10-blue)

**Production-grade AI system for real-time credit risk scoring, explainability, and agentic decision-making.**

[API Docs](#api-endpoints) • [Architecture](#architecture) • [Setup](#setup) • [Demo](#demo)

</div>

---

## 🎯 Problem Statement

Financial institutions process thousands of loan applications daily. Traditional rule-based systems are rigid, opaque, and fail to capture complex risk patterns. This system combines classical ML with modern LLM-based agentic AI to deliver:

- **Accurate risk scoring** — XGBoost trained on 1.3M real loan records
- **Explainable decisions** — SHAP values for every prediction
- **Policy-aware reasoning** — RAG over lending policy documents
- **Human-in-the-loop** — Override + audit trail for compliance
- **Production-ready** — Deployed on AWS ECS Fargate with CI/CD

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CREDIT RISK AI SYSTEM                       │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Lending Club │    │   AWS S3     │    │  XGBoost Model   │  │
│  │  1.3M Records │───▶│  Data Lake   │───▶│  AUC-ROC: 0.72   │  │
│  │  (2007-2018)  │    │  train/test  │    │  MLflow Tracked  │  │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘  │
│                                                    │             │
│  ┌─────────────────────────────────────────────────▼─────────┐  │
│  │                    FastAPI (8 Endpoints)                   │  │
│  │  /v1/predict  /v1/explain  /v1/agent  /v1/override        │  │
│  │  /v1/audit    /v1/monitor  /v1/health  /v1/model-info     │  │
│  │  API Key Auth + Input Validation + Rate Limiting           │  │
│  └──────────────────────────┬────────────────────────────────┘  │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         │                   │                   │               │
│  ┌──────▼──────┐   ┌────────▼──────┐   ┌───────▼──────┐       │
│  │  LangChain  │   │  ReAct Agent  │   │  Evaluation  │       │
│  │  RAG Layer  │   │  Groq Llama   │   │  + Monitoring│       │
│  │  Pinecone   │   │  3.3 70B      │   │  MLflow      │       │
│  │  Vector DB  │   │  4 Tools      │   │  LangSmith   │       │
│  └─────────────┘   └───────────────┘   └──────────────┘       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              AWS Infrastructure                          │   │
│  │  ECS Fargate → ECR → CloudWatch → SSM → GitHub Actions  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Model** | XGBoost 3.2 + MLflow | Credit risk scoring + experiment tracking |
| **Explainability** | SHAP | Feature importance for every prediction |
| **API** | FastAPI 0.135 | 8 versioned endpoints with auth + validation |
| **RAG** | LangChain + Pinecone | Policy document retrieval |
| **Agent** | LangGraph + Groq Llama 3.3 70B | ReAct reasoning over 4 tools |
| **Evaluation** | Custom semantic similarity + LangSmith | RAG quality measurement |
| **Monitoring** | CloudWatch + Drift Detector | Model performance tracking |
| **Storage** | AWS S3 | Data lake + model artifacts + audit logs |
| **Deployment** | Docker + AWS ECS Fargate | Containerized production deployment |
| **CI/CD** | GitHub Actions | Auto-deploy on push to main |
| **Secrets** | AWS SSM Parameter Store | Secure credential management |
| **Database** | PostgreSQL + SQLAlchemy | Agent decision audit logging (v2.1) |
| **Cache** | Redis | Async prediction caching (v2.2) |

---

## 🚀 Key Features

### 1. ML Risk Scoring
- Trained on **1.3M Lending Club loan records** (2007–2018)
- **31 features** including engineered features (loan-to-income ratio, FICO average, utilization flag)
- **AUC-ROC: 0.72** — validated with no data leakage
- Handles **19.96% class imbalance** via `scale_pos_weight`
- Full experiment tracking with **MLflow**

### 2. Explainable AI
- **SHAP values** for every prediction
- Top risk factors and protective factors returned per request
- Regulatory compliance — every decision is auditable

### 3. LangChain RAG
- Credit risk policy documents ingested into **Pinecone**
- Retrieves relevant policy context for any risk level
- Evaluated with semantic similarity (0.62) + context recall (0.62)
- Traced via **LangSmith**

### 4. ReAct Agent (Groq Llama 3.3 70B)
- **4 specialized tools**: PredictRisk, RetrievePolicy, ExplainDecision, RetrieveSimilarCases
- Agent reasons step-by-step before making final recommendation
- Returns fully justified credit decision with policy citations

### 5. Human-in-the-Loop (HITL)
- Manual override endpoint for underwriter review
- Full audit trail stored in S3
- Compliant with ECOA and FCRA requirements

### 6. Model Drift Detection
- Monitors AUC-ROC against baseline
- CloudWatch metrics for real-time alerting
- CRITICAL alert if AUC drops below 0.70 (policy threshold)

---

## 📡 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/v1/health` | ❌ | ECS health check |
| `GET` | `/v1/model-info` | ❌ | Model metadata + metrics |
| `POST` | `/v1/predict` | ✅ | Risk score + decision |
| `POST` | `/v1/explain` | ✅ | SHAP feature importance |
| `POST` | `/v1/agent` | ✅ | Full ReAct agent analysis |
| `POST` | `/v1/override` | ✅ | HITL manual override |
| `GET` | `/v1/audit` | ✅ | Override audit log |
| `GET` | `/v1/monitor` | ✅ | Model drift report |

All authenticated endpoints require `X-API-Key` header.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **AUC-ROC** | 0.7198 |
| **PR-AUC** | 0.3875 |
| **Training samples** | 1,076,248 |
| **Test samples** | 269,062 |
| **Default rate** | 19.96% |
| **Features** | 31 |
| **RAG semantic similarity** | 0.6168 |
| **RAG context recall** | 0.6168 |
| **Model drift** | 0.01% (HEALTHY) |

---

## 🗂️ Project Structure

```
credit-risk-ai/
├── src/
│   ├── pipeline/
│   │   └── data_pipeline.py      # S3 data ingestion + feature engineering
│   ├── model/
│   │   └── train.py              # XGBoost training + MLflow tracking
│   ├── api/
│   │   └── main.py               # FastAPI — 8 endpoints
│   ├── rag/
│   │   ├── ingest.py             # Pinecone document ingestion
│   │   └── retriever.py          # LangChain RAG chain
│   ├── agent/
│   │   └── agent.py              # LangGraph ReAct agent + 4 tools
│   ├── evaluation/
│   │   └── evaluate_rag.py       # RAG evaluation pipeline
│   └── monitoring/
│       └── drift_detector.py     # Model drift detection + CloudWatch
├── policies/
│   └── credit_risk_policy.txt    # Credit risk policy document
├── docker/
│   ├── docker-compose.yml        # Local development
│   └── task-definition.json      # AWS ECS task definition
├── .github/
│   └── workflows/
│       └── deploy.yml            # GitHub Actions CI/CD
├── Dockerfile                    # Container definition
├── requirements.txt
└── .env.example
```

---

## 🛠️ Setup

### Prerequisites
- Python 3.12+
- Docker Desktop
- AWS CLI configured
- API keys: Groq, Pinecone, LangSmith

### 1. Clone the repository
```bash
git clone https://github.com/prnav1923/credit-risk-ai.git
cd credit-risk-ai
```

### 2. Create virtual environment
```bash
python3 -m venv .credai
source .credai/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
# Fill in your API keys
```

### 4. Run locally
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 5. Run with Docker
```bash
docker-compose -f docker/docker-compose.yml up
```

### 6. Test the API
```bash
# Health check
curl http://localhost:8000/v1/health

# Predict risk
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "loan_amnt": 10000,
    "int_rate": 12.5,
    "annual_inc": 60000,
    "dti": 18.5,
    "fico_range_low": 680,
    "fico_range_high": 684,
    "grade": "B",
    "sub_grade": "B3",
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "application_type": "Individual",
    "initial_list_status": "w",
    "installment": 350.0,
    "delinq_2yrs": 0,
    "open_acc": 10,
    "pub_rec": 0,
    "revol_bal": 15000,
    "revol_util": 45.0,
    "total_acc": 25,
    "emp_length": 5,
    "mort_acc": 2,
    "pub_rec_bankruptcies": 0,
    "num_actv_bc_tl": 4,
    "bc_util": 50.0,
    "percent_bc_gt_75": 25.0,
    "avg_cur_bal": 8000
  }'
```

---

## 🔄 CI/CD Pipeline

Every push to `main` triggers:

```
git push origin main
        ↓
GitHub Actions
        ↓
Build Docker image (linux/amd64)
        ↓
Push to AWS ECR
        ↓
Update ECS task definition
        ↓
Deploy to ECS Fargate
        ↓
Live API updated ✅
```

---

## 🧪 Running Evaluations

```bash
# RAG evaluation
python src/evaluation/evaluate_rag.py

# Model drift detection
python src/monitoring/drift_detector.py

# Or via API
curl http://localhost:8000/v1/monitor \
  -H "X-API-Key: your-api-key"
```

---

## 🔐 Security

- API key authentication on all business endpoints
- Secrets stored in AWS SSM Parameter Store
- Input validation with Pydantic (range checks, enum validation)
- No credentials in codebase or Docker image
- Audit trail for all human overrides

---

## 📈 Roadmap

- [x] v1.0 — Core system (ML + RAG + Agent + API + Deployment)
- [ ] v2.0 — Fraud Detection + Multi-Agent + MCP Server + Streamlit UI
- [ ] v2.1 — PostgreSQL + SQLAlchemy for agent decision audit logging
- [ ] v2.2 — Redis caching for async prediction endpoints
- [ ] v3.0 — Kafka streaming + Real-time scoring pipeline
---

## 👤 Author

**Pranav** — Data Analyst → AI Engineer  
[GitHub](https://github.com/prnav1923) • [LinkedIn](https://www.linkedin.com/in/pranav-kumar-1b6843203/)

---

<div align="center">
<i>Built with production-grade engineering practices — not just a demo.</i>
</div>
