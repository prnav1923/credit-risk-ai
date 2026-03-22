# 🏦 Credit Risk Assessment AI System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green?logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)
![LangChain](https://img.shields.io/badge/LangChain-1.2-purple)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1-purple)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-red)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-blue)
![AWS](https://img.shields.io/badge/AWS-ECS%20Fargate-FF9900?logo=amazonaws)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions)
![MLflow](https://img.shields.io/badge/MLflow-3.10-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue?logo=postgresql)
![Redis](https://img.shields.io/badge/Redis-Caching-red?logo=redis)
![Kafka](https://img.shields.io/badge/Kafka-Streaming-black?logo=apachekafka)
![MCP](https://img.shields.io/badge/MCP-Claude%20Desktop-orange)

**Production-grade AI system combining classical ML, fraud detection, multi-agent reasoning, real-time streaming, and Model Context Protocol (MCP) for end-to-end credit risk assessment.**

[Architecture](#architecture) • [API Endpoints](#api-endpoints) • [MCP Integration](#mcp-integration) • [Setup](#setup) • [Demo](#demo)

</div>

---

## 🎯 Problem Statement

Financial institutions process thousands of loan applications daily. Traditional rule-based systems are rigid, opaque, and fail to capture complex risk patterns. This system solves three core problems:

1. **Credit Risk** — Is this borrower likely to default?
2. **Fraud Detection** — Does this application show signs of fraud?
3. **Explainability** — Why did the system make this decision?

It combines classical ML with modern LLM-based multi-agent AI, real-time Kafka streaming, PostgreSQL persistence, Redis caching, and exposes everything via a Model Context Protocol (MCP) server — enabling any MCP-compatible client (like Claude Desktop) to call the system's tools in natural language.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  CREDIT RISK ASSESSMENT AI SYSTEM v2.0                   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        DATA LAYER                                │   │
│  │  Lending Club (1.3M records) → S3 Data Lake                     │   │
│  │  Feature Engineering (31 features, zero leakage validated)      │   │
│  │  Kafka Real-time Streaming → Consumer → Score → PostgreSQL      │   │
│  └──────────────────────────┬───────────────────────────────────────┘   │
│                             │                                            │
│  ┌──────────────────────────▼───────────────────────────────────────┐   │
│  │                        ML LAYER                                  │   │
│  │  ┌──────────────────────────┐  ┌──────────────────────────────┐ │   │
│  │  │    Credit Risk Model     │  │   Fraud Detection Ensemble   │ │   │
│  │  │    XGBoost Classifier    │  │  Isolation Forest + XGBOD    │ │   │
│  │  │    AUC-ROC: 0.7198       │  │  AUC-ROC: 0.8841            │ │   │
│  │  │    MLflow Tracked        │  │  Fraud Rate: 7.31%           │ │   │
│  │  └──────────────────────────┘  └──────────────────────────────┘ │   │
│  └──────────────────────────┬───────────────────────────────────────┘   │
│                             │                                            │
│  ┌──────────────────────────▼───────────────────────────────────────┐   │
│  │                    INTELLIGENCE LAYER                            │   │
│  │  ┌───────────────────┐  ┌──────────────────────────────────────┐│   │
│  │  │  LangChain RAG    │  │  Multi-Agent (Groq Llama 3.3 70B)   ││   │
│  │  │  Pinecone Vector  │  │  ┌────────────────────────────────┐ ││   │
│  │  │  Policy Docs      │  │  │ Risk Agent                     │ ││   │
│  │  │  Sem Sim: 0.62    │  │  │ PredictRisk+ExplainDecision    │ ││   │
│  │  └───────────────────┘  │  │ +DetectFraud                   │ ││   │
│  │                         │  ├────────────────────────────────┤ ││   │
│  │                         │  │ Compliance Agent               │ ││   │
│  │                         │  │ RetrievePolicy+CheckECOA/FCRA  │ ││   │
│  │                         │  ├────────────────────────────────┤ ││   │
│  │                         │  │ Decision Agent                 │ ││   │
│  │                         │  │ Synthesize Final Decision      │ ││   │
│  │                         │  └────────────────────────────────┘ ││   │
│  │                         └──────────────────────────────────────┘│   │
│  └──────────────────────────┬───────────────────────────────────────┘   │
│                             │                                            │
│  ┌──────────────────────────▼───────────────────────────────────────┐   │
│  │              FastAPI — 13 Versioned Endpoints                    │   │
│  │  /v1/assess  /v1/predict  /v1/fraud  /v1/explain                │   │
│  │  /v1/agent   /v1/multi-agent  /v1/override  /v1/audit           │   │
│  │  /v1/monitor /v1/cache  /v1/predictions  /v1/agent-decisions    │   │
│  │  /v1/health  /v1/model-info                                     │   │
│  │  API Key Auth + Pydantic Validation + Redis Caching             │   │
│  └──────────────┬───────────────────────┬────────────────────────┬─┘   │
│                 │                       │                        │      │
│  ┌──────────────▼──────┐  ┌────────────▼──────┐  ┌─────────────▼─┐   │
│  │   PostgreSQL        │  │   Redis Cache     │  │  MCP Server   │   │
│  │   4 Tables          │  │   Predictions     │  │   9 Tools     │   │
│  │   predictions       │  │   1hr TTL         │  │   Claude      │   │
│  │   agent_decisions   │  └───────────────────┘  │   Desktop     │   │
│  │   overrides         │                         └───────────────┘   │
│  │   drift_reports     │                                              │
│  └─────────────────────┘                                              │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              AWS INFRASTRUCTURE                                  │   │
│  │  ECS Fargate → ECR → CloudWatch → SSM → GitHub Actions CI/CD   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Credit Risk ML** | XGBoost 3.2 + MLflow | Risk scoring + experiment tracking |
| **Fraud Detection** | Isolation Forest + XGBOD ensemble | Anomaly detection |
| **Explainability** | SHAP | Feature importance per prediction |
| **API** | FastAPI 0.135 | 13 versioned endpoints + auth + validation |
| **RAG** | LangChain + Pinecone | Policy document retrieval |
| **Single Agent** | LangGraph + Groq Llama 3.3 70B | ReAct reasoning over 4 tools |
| **Multi-Agent** | LangGraph StateGraph | Risk + Compliance + Decision agents |
| **MCP Server** | MCP SDK | 9 tools for Claude Desktop |
| **Streaming** | Apache Kafka | Real-time loan application pipeline |
| **Database** | PostgreSQL + SQLAlchemy | Predictions + decisions + audit logs |
| **Cache** | Redis | Prediction result caching (1hr TTL) |
| **Evaluation** | Sentence-transformers + LangSmith | RAG quality measurement |
| **Monitoring** | CloudWatch + Drift Detector | Model performance tracking |
| **Storage** | AWS S3 | Data lake + models + audit logs |
| **Deployment** | Docker + AWS ECS Fargate | Containerized production |
| **CI/CD** | GitHub Actions | Auto-deploy on push to main |
| **Frontend** | Streamlit | 5-page interactive UI |
| **Secrets** | AWS SSM Parameter Store | Secure credential management |

---

## 🚀 Key Features

### 1. Dual ML System — Credit Risk + Fraud Detection

**Credit Risk (XGBoost):**
- Trained on **1.3M Lending Club records** (2007–2018)
- **31 features** including engineered features (loan-to-income, FICO avg, utilization flag)
- **AUC-ROC: 0.7198** — zero data leakage validated via correlation analysis
- **19.96% class imbalance** handled via `scale_pos_weight`

**Fraud Detection (Isolation Forest + XGBOD Ensemble):**
- **Isolation Forest** — unsupervised anomaly detection, no fraud labels required
- **XGBOD** — supervised ensemble using proxy fraud labels from behavioral indicators
- **AUC-ROC: 0.8841** — strong detection on 7.31% fraud rate
- Detects: high utilization, delinquencies, public records, bankruptcy, suspicious patterns

**Combined `/v1/assess` response (low risk):**
```json
{
  "credit_risk_score": 0.3638,
  "credit_decision": "REVIEW",
  "fraud_score": 0.4006,
  "fraud_flag": false,
  "fraud_risk": "MEDIUM",
  "fraud_indicators": [],
  "combined_decision": "REVIEW",
  "combined_risk": "MANUAL REVIEW REQUIRED"
}
```

**Combined `/v1/assess` response (high risk + fraud):**
```json
{
  "credit_risk_score": 0.8823,
  "credit_decision": "DECLINE",
  "fraud_score": 0.8285,
  "fraud_flag": true,
  "fraud_risk": "HIGH",
  "fraud_indicators": [
    "High revolving utilization (>80%)",
    "Multiple delinquencies in last 2 years",
    "Public records present",
    "Bankruptcy history",
    "High loan-to-income ratio (>0.4)",
    "High DTI ratio (>40%)"
  ],
  "combined_decision": "DECLINE",
  "combined_risk": "FRAUD DETECTED"
}
```

### 2. Explainable AI (SHAP)
- SHAP values for every prediction
- Top risk and protective factors per request
- Regulatory compliance — every decision is fully auditable

### 3. Multi-Agent Workflow (3 Agents)
```
Risk Agent → Compliance Agent → Decision Agent
```
- **Risk Agent**: PredictRisk + ExplainDecision + DetectFraud
- **Compliance Agent**: RetrievePolicy + CheckRegulations (ECOA/FCRA)
- **Decision Agent**: Synthesizes final recommendation with conditions

### 4. Real-Time Kafka Streaming
```
Loan Applications → Kafka Topic (3 partitions)
        ↓
Consumer scores each application via /v1/assess
        ↓
Results logged to PostgreSQL
        ↓
Real-time decision stats
```

### 5. PostgreSQL Persistence (4 Tables)
- `predictions` — every scored application
- `agent_decisions` — single + multi-agent reports
- `overrides` — HITL reviewer decisions
- `drift_reports` — model performance history

### 6. Redis Caching
- `/v1/predict` results cached for 1 hour
- Cache key = MD5 hash of application payload
- Cache stats via `/v1/cache`

### 7. Human-in-the-Loop (HITL)
- Manual override endpoint for underwriter review
- Dual logging — PostgreSQL + S3
- ECOA and FCRA compliant audit trail

### 8. Model Drift Detection
- AUC-ROC monitored vs baseline (0.7198)
- CloudWatch metrics + PostgreSQL logging
- CRITICAL alert if AUC < 0.70

### 9. MCP Server (9 Tools)
- Full system exposed via Model Context Protocol
- Complete HITL workflow in Claude Desktop
- Natural language access to all capabilities

---

## 📡 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/v1/health` | ❌ | ECS health check |
| `GET` | `/v1/model-info` | ❌ | Model metadata + metrics |
| `POST` | `/v1/assess` | ✅ | ⭐ Combined credit + fraud |
| `POST` | `/v1/predict` | ✅ | Credit risk only (Redis cached) |
| `POST` | `/v1/fraud` | ✅ | Fraud detection only |
| `POST` | `/v1/explain` | ✅ | SHAP feature importance |
| `POST` | `/v1/agent` | ✅ | Single ReAct agent |
| `POST` | `/v1/multi-agent` | ✅ | ⭐ 3-agent workflow |
| `POST` | `/v1/override` | ✅ | HITL manual override |
| `GET` | `/v1/audit` | ✅ | S3 override audit log |
| `GET` | `/v1/monitor` | ✅ | Model drift report |
| `GET` | `/v1/cache` | ✅ | Redis cache stats |
| `GET` | `/v1/predictions` | ✅ | PostgreSQL predictions history |
| `GET` | `/v1/agent-decisions` | ✅ | PostgreSQL agent decisions |

---

## 🔌 MCP Integration

### MCP Tools (9 Total)

| Tool | Description |
|------|-------------|
| `assess_credit_risk` | Credit risk score + decision |
| `explain_credit_decision` | SHAP risk + protective factors |
| `query_credit_policy` | Pinecone policy search |
| `run_full_agent_analysis` | Single ReAct agent |
| `assess_combined` | Combined credit + fraud assessment |
| `run_multi_agent_analysis` | 3-agent workflow |
| `override_credit_decision` | HITL override → S3 audit |
| `get_audit_log` | Full compliance history |
| `get_model_info` | Model metadata + AUC scores |

### Setup Claude Desktop

Add to `~/Library/Application\ Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "credit-risk-ai": {
      "command": "/path/to/credit-risk-ai/.credai/bin/python",
      "args": ["/path/to/credit-risk-ai/src/mcp_server.py"],
      "env": {
        "API_BASE_URL": "http://127.0.0.1:8000",
        "API_KEY": "your-api-key",
        "PINECONE_API_KEY": "your-pinecone-key",
        "PINECONE_INDEX_NAME": "credit-risk-policy"
      }
    }
  }
}
```

### Full HITL Demo in Claude Desktop

```
Step 1: "Use assess_combined tool for this application: [data]"
        → Credit: 0.3638 REVIEW | Fraud: 0.4006 MEDIUM

Step 2: "Explain using explain_credit_decision tool"
        → SHAP: sub_grade -0.338 (protective), home_ownership +0.063 (risk)

Step 3: "Query policy using query_credit_policy: when is override required?"
        → Policy: REVIEW band 0.3-0.6, human discretion for edge cases

Step 4: "Override to APPROVE as underwriter_pranav — stable employment"
        → Logged to S3 + PostgreSQL audit trail

Step 5: "Show audit log using get_audit_log"
        → Full compliance record with reviewer ID + timestamp

Step 6: "Run run_multi_agent_analysis"
        → 3 agents: Risk → Compliance → Decision → APPROVE with conditions
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Credit Risk AUC-ROC** | 0.7198 |
| **Credit Risk PR-AUC** | 0.3875 |
| **Fraud Detection AUC-ROC** | 0.8841 |
| **Fraud Rate (training)** | 7.31% |
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
│   │   └── data_pipeline.py        # S3 ingestion + feature engineering
│   ├── model/
│   │   ├── train.py                # XGBoost + MLflow
│   │   └── fraud_detector.py       # Isolation Forest + XGBOD
│   ├── api/
│   │   └── main.py                 # FastAPI — 13 endpoints
│   ├── rag/
│   │   ├── ingest.py               # Pinecone ingestion
│   │   └── retriever.py            # LangChain RAG chain
│   ├── agent/
│   │   ├── agent.py                # Single ReAct agent
│   │   └── multi_agent.py          # 3-agent LangGraph workflow
│   ├── evaluation/
│   │   └── evaluate_rag.py         # RAG evaluation pipeline
│   ├── monitoring/
│   │   └── drift_detector.py       # Drift detection + CloudWatch
│   ├── database.py                 # PostgreSQL models + sessions
│   ├── cache.py                    # Redis caching utilities
│   ├── kafka_producer.py           # Loan application producer
│   ├── kafka_consumer.py           # Real-time scoring consumer
│   └── mcp_server.py               # MCP server — 9 tools
├── streamlit_app.py                # 5-page Streamlit UI
├── policies/
│   └── credit_risk_policy.txt
├── docker/
│   ├── docker-compose.yml
│   └── task-definition.json
├── .github/workflows/deploy.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 🛠️ Setup

### Prerequisites
- Python 3.12+
- Docker Desktop
- PostgreSQL 15
- Redis
- Apache Kafka
- AWS CLI configured
- API keys: Groq, Pinecone, LangSmith

### 1. Clone and Install
```bash
git clone https://github.com/prnav1923/credit-risk-ai.git
cd credit-risk-ai
python3 -m venv .credai
source .credai/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Fill in all API keys
```

### 3. Setup Database
```bash
createdb credit_risk_db
python src/database.py
```

### 4. Start Services
```bash
brew services start postgresql@15
brew services start redis
brew services start kafka
```

### 5. Run Pipeline + Training
```bash
python src/pipeline/data_pipeline.py
python src/model/train.py
python src/model/fraud_detector.py
python src/rag/ingest.py
```

### 6. Start API
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 7. Start Streamlit
```bash
streamlit run streamlit_app.py
```

### 8. Docker
```bash
docker-compose -f docker/docker-compose.yml up
```

---

## 🧪 Quick Tests

```bash
# Health
curl http://localhost:8000/v1/health

# Combined Assessment
curl -X POST http://localhost:8000/v1/assess \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"loan_amnt":10000,"int_rate":12.5,"installment":350,"annual_inc":60000,"dti":18.5,"delinq_2yrs":0,"fico_range_low":680,"fico_range_high":684,"open_acc":10,"pub_rec":0,"revol_bal":15000,"revol_util":45,"total_acc":25,"emp_length":5,"mort_acc":2,"pub_rec_bankruptcies":0,"num_actv_bc_tl":4,"bc_util":50,"percent_bc_gt_75":25,"avg_cur_bal":8000,"home_ownership":"RENT","verification_status":"Verified","purpose":"debt_consolidation","grade":"B","sub_grade":"B3","initial_list_status":"w","application_type":"Individual"}'

# Kafka Streaming
python src/kafka_consumer.py  # Terminal 1
python src/kafka_producer.py  # Terminal 2

# Drift Detection
python src/monitoring/drift_detector.py

# RAG Evaluation
python src/evaluation/evaluate_rag.py
```

---

## 🖥️ Streamlit UI (5 Pages)

| Page | Description |
|------|-------------|
| 🎯 Single Assessment | Credit + fraud assessment with gauges + SHAP chart |
| 🤖 Multi-Agent Analysis | 3-agent workflow with architecture diagram |
| 📊 Model Info | AUC scores, drift status, RAG metrics |
| 📋 Audit Log | HITL override history + submit new override |
| 📈 Predictions History | All scored applications with charts from PostgreSQL |

---

## 🔄 CI/CD Pipeline

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
Deploy to ECS Fargate ✅
```

---

## 🔐 Security

- API key authentication on all business endpoints
- Secrets in AWS SSM Parameter Store
- Pydantic input validation (range checks, enum validation)
- No credentials in code or Docker image
- Dual audit trail: PostgreSQL + S3 (ECOA/FCRA compliant)

---

## 📈 Roadmap

- [x] v1.0 — Core system (ML + RAG + Agent + API + ECS + CI/CD)
- [x] v1.1 — Fraud Detection (Isolation Forest + XGBOD, AUC 0.88)
- [x] v1.2 — MCP Server (7 tools, Claude Desktop, full HITL demo)
- [x] v2.0 — Multi-Agent (Risk + Compliance + Decision) + PostgreSQL + Redis + Kafka + Streamlit + 9 MCP tools
- [ ] v3.0 — Fine-tuning domain-specific LLM on credit risk Q&A
- [ ] v3.1 — Real-time dashboards with Grafana + Prometheus

---

## 👤 Author

**Pranav** — Data Analyst → AI Engineer
[GitHub](https://github.com/prnav1923) • [LinkedIn](https://linkedin.com/in/yourprofile)

---

<div align="center">
<i>Built with production-grade engineering — not just a demo.</i>
</div>
