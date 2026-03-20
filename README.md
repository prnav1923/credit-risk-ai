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
![MCP](https://img.shields.io/badge/MCP-Claude%20Desktop-orange)

**Production-grade AI system combining classical ML, fraud detection, LLM-based agentic reasoning, and Model Context Protocol (MCP) for real-time credit risk assessment.**

[Architecture](#architecture) • [API Endpoints](#api-endpoints) • [MCP Integration](#mcp-integration) • [Setup](#setup) • [Demo](#demo)

</div>

---

## 🎯 Problem Statement

Financial institutions process thousands of loan applications daily. Traditional rule-based systems are rigid, opaque, and fail to capture complex risk patterns. This system solves three core problems:

1. **Credit Risk** — Is this borrower likely to default?
2. **Fraud Detection** — Does this application show signs of fraud?
3. **Explainability** — Why did the system make this decision?

It combines classical ML with modern LLM-based agentic AI and exposes everything via a Model Context Protocol (MCP) server — enabling any MCP-compatible client (like Claude Desktop) to directly call the system's tools in natural language.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   CREDIT RISK ASSESSMENT AI SYSTEM v1.2             │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    DATA LAYER                               │   │
│  │  Lending Club (1.3M records) → S3 Data Lake                │   │
│  │  Feature Engineering (31 features, no leakage)             │   │
│  └─────────────────────┬───────────────────────────────────────┘   │
│                        │                                           │
│  ┌─────────────────────▼───────────────────────────────────────┐   │
│  │                    ML LAYER                                 │   │
│  │  ┌─────────────────────┐  ┌──────────────────────────────┐ │   │
│  │  │   Credit Risk Model  │  │    Fraud Detection Ensemble  │ │   │
│  │  │   XGBoost Classifier │  │  Isolation Forest + XGBOD   │ │   │
│  │  │   AUC-ROC: 0.7198    │  │  AUC-ROC: 0.8841            │ │   │
│  │  │   MLflow Tracked     │  │  7.31% fraud rate           │ │   │
│  │  └─────────────────────┘  └──────────────────────────────┘ │   │
│  └─────────────────────┬───────────────────────────────────────┘   │
│                        │                                           │
│  ┌─────────────────────▼───────────────────────────────────────┐   │
│  │                  INTELLIGENCE LAYER                         │   │
│  │  ┌──────────────────┐  ┌──────────────────────────────────┐│   │
│  │  │  LangChain RAG   │  │    ReAct Agent (Groq Llama 3.3) ││   │
│  │  │  Pinecone Vector │  │    4 Tools: PredictRisk,        ││   │
│  │  │  Policy Docs     │  │    RetrievePolicy,              ││   │
│  │  │  Sem Sim: 0.62   │  │    ExplainDecision,             ││   │
│  │  └──────────────────┘  │    RetrieveSimilarCases         ││   │
│  │                        └──────────────────────────────────┘│   │
│  └─────────────────────┬───────────────────────────────────────┘   │
│                        │                                           │
│  ┌─────────────────────▼───────────────────────────────────────┐   │
│  │              FastAPI — 10 Versioned Endpoints               │   │
│  │  /v1/assess  /v1/predict  /v1/fraud  /v1/explain            │   │
│  │  /v1/agent   /v1/override /v1/audit  /v1/monitor            │   │
│  │  /v1/health  /v1/model-info                                 │   │
│  │  API Key Auth + Input Validation + Pydantic Schemas         │   │
│  └─────────────────────┬───────────────────────────────────────┘   │
│                        │                                           │
│  ┌─────────────────────▼───────────────────────────────────────┐   │
│  │                   MCP SERVER                                │   │
│  │  7 Tools exposed for Claude Desktop integration             │   │
│  │  assess_credit_risk • explain_credit_decision               │   │
│  │  query_credit_policy • run_full_agent_analysis              │   │
│  │  override_credit_decision • get_audit_log • get_model_info  │   │
│  └─────────────────────┬───────────────────────────────────────┘   │
│                        │                                           │
│  ┌─────────────────────▼───────────────────────────────────────┐   │
│  │              AWS INFRASTRUCTURE                             │   │
│  │  ECS Fargate → ECR → CloudWatch → SSM → GitHub Actions     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Credit Risk ML** | XGBoost 3.2 + MLflow | Risk scoring + experiment tracking |
| **Fraud Detection** | Isolation Forest + XGBOD ensemble | Anomaly detection |
| **Explainability** | SHAP | Feature importance per prediction |
| **API** | FastAPI 0.135 | 10 versioned endpoints + auth + validation |
| **RAG** | LangChain + Pinecone | Policy document retrieval |
| **Agent** | LangGraph + Groq Llama 3.3 70B | ReAct reasoning over 4 tools |
| **MCP Server** | MCP SDK | 7 tools for Claude Desktop |
| **Evaluation** | Sentence-transformers + LangSmith | RAG quality measurement |
| **Monitoring** | CloudWatch + Drift Detector | Model performance tracking |
| **Storage** | AWS S3 | Data lake + models + audit logs |
| **Deployment** | Docker + AWS ECS Fargate | Containerized production |
| **CI/CD** | GitHub Actions | Auto-deploy on push to main |
| **Secrets** | AWS SSM Parameter Store | Secure credential management |

---

## 🚀 Key Features

### 1. Dual ML System — Credit Risk + Fraud Detection

**Credit Risk (XGBoost):**
- Trained on **1.3M Lending Club loan records** (2007–2018)
- **31 features** including engineered features (loan-to-income, FICO average, utilization flag)
- **AUC-ROC: 0.7198** — validated with zero data leakage
- Handles **19.96% class imbalance** via `scale_pos_weight`
- Full experiment tracking with **MLflow**

**Fraud Detection (Isolation Forest + XGBOD Ensemble):**
- **Isolation Forest** — unsupervised anomaly detection, no fraud labels required
- **XGBOD** — supervised ensemble using proxy fraud labels from behavioral indicators
- **AUC-ROC: 0.8841** — strong detection performance
- **7.31% fraud rate** identified in training data
- Detects: high utilization, multiple delinquencies, public records, bankruptcy, suspicious loan-to-income

**Combined `/v1/assess` response:**
```json
{
  "credit_risk_score": 0.3638,
  "credit_decision": "REVIEW",
  "credit_risk_level": "MEDIUM RISK",
  "fraud_score": 0.4006,
  "fraud_flag": false,
  "fraud_risk": "MEDIUM",
  "fraud_indicators": [],
  "combined_decision": "REVIEW",
  "combined_risk": "MANUAL REVIEW REQUIRED",
  "message": "REVIEW — Credit Risk: 0.3638, Fraud Risk: MEDIUM"
}
```

**High-risk fraud case response:**
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
- Top risk factors and protective factors per request
- Regulatory compliance — every decision is fully auditable
```json
{
  "top_risk_factors": [
    {"feature": "home_ownership", "impact": 0.0629},
    {"feature": "fico_range_low", "impact": 0.0431}
  ],
  "top_protective_factors": [
    {"feature": "sub_grade", "impact": -0.3377},
    {"feature": "grade", "impact": -0.1670}
  ]
}
```

### 3. LangChain RAG (Pinecone)
- Credit risk policy documents ingested into Pinecone vector store
- Retrieves relevant policy context for any risk level
- Semantic similarity: **0.6168** | Context recall: **0.6168**
- All traces logged to **LangSmith**

### 4. ReAct Agent (Groq Llama 3.3 70B)
- **4 tools**: PredictRisk, RetrievePolicy, ExplainDecision, RetrieveSimilarCases
- True ReAct loop — agent reasons which tools to call and in what order
- Returns fully justified decision with policy citations

### 5. Human-in-the-Loop (HITL)
- Manual override for underwriter review
- Full audit trail in S3 — reviewer ID, reason, timestamp
- ECOA and FCRA compliant
- Retrievable via `/v1/audit`

### 6. Model Drift Detection
- Monitors AUC-ROC vs baseline (0.7198)
- CloudWatch metrics
- CRITICAL alert if AUC < 0.70 (policy threshold)
- Live via `/v1/monitor`

### 7. MCP Server
- 7 tools via Model Context Protocol
- Full HITL workflow in Claude Desktop
- Natural language access to all system capabilities

---

## 📡 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/v1/health` | ❌ | ECS health check |
| `GET` | `/v1/model-info` | ❌ | Model metadata + metrics |
| `POST` | `/v1/assess` | ✅ | ⭐ Combined credit risk + fraud |
| `POST` | `/v1/predict` | ✅ | Credit risk score only |
| `POST` | `/v1/fraud` | ✅ | Fraud detection only |
| `POST` | `/v1/explain` | ✅ | SHAP feature importance |
| `POST` | `/v1/agent` | ✅ | Full ReAct agent analysis |
| `POST` | `/v1/override` | ✅ | HITL manual override |
| `GET` | `/v1/audit` | ✅ | Override audit log |
| `GET` | `/v1/monitor` | ✅ | Model drift report |

All authenticated endpoints require `X-API-Key` header.

---

## 🔌 MCP Integration

This system is exposed as an **MCP server**, enabling Claude Desktop to call credit risk tools in natural language.

### MCP Tools

| Tool | Description |
|------|-------------|
| `assess_credit_risk` | Score a loan — risk score, decision, risk level |
| `explain_credit_decision` | SHAP explanation — top risk and protective factors |
| `query_credit_policy` | Semantic search over Pinecone policy documents |
| `run_full_agent_analysis` | Complete ReAct agent analysis |
| `override_credit_decision` | HITL override — logs to S3 audit trail |
| `get_audit_log` | Full override history for compliance |
| `get_model_info` | Model metadata and AUC scores |

### Setup

**1. Configure Claude Desktop:**

Add to `~/Library/Application\ Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "credit-risk-ai": {
      "command": "/Users/yourname/Code/credit-risk-ai/.credai/bin/python",
      "args": ["/Users/yourname/Code/credit-risk-ai/src/mcp_server.py"],
      "env": {
        "API_BASE_URL": "http://localhost:8000",
        "API_KEY": "your-api-key",
        "PINECONE_API_KEY": "your-pinecone-key",
        "PINECONE_INDEX_NAME": "credit-risk-policy"
      }
    }
  }
}
```

**2. Start API + restart Claude Desktop**

**3. Verify** — look for 🔨 tools icon in Claude Desktop

### Full HITL Demo in Claude Desktop

**Step 1 — Assess:**
```
Assess the credit risk for the sample application using assess_credit_risk tool
```
→ Returns: Risk score 0.364, decision REVIEW

**Step 2 — Explain:**
```
Explain what factors drove this decision using explain_credit_decision tool
```
→ Returns: SHAP values, sub_grade strongest protective factor (-0.338)

**Step 3 — Query Policy:**
```
Using query_credit_policy tool, what does our policy say about 
REVIEW decisions and when human override is required?
```
→ Returns: Policy retrieval — manual review criteria for score 0.3-0.6 band

**Step 4 — Override:**
```
As underwriter_pranav, override to APPROVE using override_credit_decision tool.
Reason: Borrower has 10 years stable employment and decreasing revolving balance.
```
→ Returns: Override confirmed, logged to S3 audit trail

**Step 5 — Audit:**
```
Show me the full audit log using get_audit_log tool
```
→ Returns: Full compliance log with reviewer ID, timestamp, decisions

**Step 6 — Full Analysis:**
```
Run complete analysis using run_full_agent_analysis tool
```
→ Returns: Groq Llama 3.3 70B reasoning through all 4 tools

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
│   │   └── data_pipeline.py       # S3 data ingestion + feature engineering
│   ├── model/
│   │   ├── train.py               # XGBoost + MLflow
│   │   └── fraud_detector.py      # Isolation Forest + XGBOD ensemble
│   ├── api/
│   │   └── main.py                # FastAPI — 10 endpoints
│   ├── rag/
│   │   ├── ingest.py              # Pinecone ingestion
│   │   └── retriever.py           # LangChain RAG chain
│   ├── agent/
│   │   └── agent.py               # LangGraph ReAct + 4 tools
│   ├── evaluation/
│   │   └── evaluate_rag.py        # RAG evaluation
│   ├── monitoring/
│   │   └── drift_detector.py      # Drift detection + CloudWatch
│   └── mcp_server.py              # MCP server — 7 tools
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

```bash
# Clone
git clone https://github.com/prnav1923/credit-risk-ai.git
cd credit-risk-ai

# Install
python3 -m venv .credai
source .credai/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env  # fill in API keys

# Pipeline
python src/pipeline/data_pipeline.py

# Train
python src/model/train.py
python src/model/fraud_detector.py

# Ingest policies
python src/rag/ingest.py

# Start API
uvicorn src.api.main:app --reload --port 8000

# Or Docker
docker-compose -f docker/docker-compose.yml up
```

---

## 🧪 Quick Tests

```bash
# Health
curl http://localhost:8000/v1/health

# Combined assessment
curl -X POST http://localhost:8000/v1/assess \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"loan_amnt":10000,"int_rate":12.5,"installment":350,"annual_inc":60000,"dti":18.5,"delinq_2yrs":0,"fico_range_low":680,"fico_range_high":684,"open_acc":10,"pub_rec":0,"revol_bal":15000,"revol_util":45,"total_acc":25,"emp_length":5,"mort_acc":2,"pub_rec_bankruptcies":0,"num_actv_bc_tl":4,"bc_util":50,"percent_bc_gt_75":25,"avg_cur_bal":8000,"home_ownership":"RENT","verification_status":"Verified","purpose":"debt_consolidation","grade":"B","sub_grade":"B3","initial_list_status":"w","application_type":"Individual"}'

# Drift monitor
curl http://localhost:8000/v1/monitor -H "X-API-Key: your-key"

# RAG evaluation
python src/evaluation/evaluate_rag.py
```

---

## 🔄 CI/CD

Every push to `main`:
```
git push → GitHub Actions → Docker build (linux/amd64) → ECR → ECS Fargate ✅
```

---

## 🔐 Security

- API key auth on all business endpoints
- Secrets in AWS SSM Parameter Store
- Pydantic input validation (range checks, enum validation)
- No credentials in code or Docker image
- Full S3 audit trail (ECOA/FCRA compliant)

---

## 📈 Roadmap

- [x] v1.0 — Core system (ML + RAG + Agent + API + ECS + CI/CD)
- [x] v1.1 — Fraud Detection (Isolation Forest + XGBOD, AUC 0.88)
- [x] v1.2 — MCP Server (7 tools, Claude Desktop, full HITL demo)
- [ ] v2.0 — Multi-Agent (Risk + Fraud + Compliance + Decision agents)
- [ ] v2.1 — Streamlit frontend
- [ ] v2.2 — PostgreSQL + SQLAlchemy audit logging
- [ ] v2.3 — Redis async caching
- [ ] v3.0 — Kafka streaming + real-time pipeline

---

## 👤 Author

**Pranav** — Data Analyst → AI Engineer
[GitHub](https://github.com/prnav1923) • [LinkedIn](https://linkedin.com/in/yourprofile)

---

<div align="center">
<i>Built with production-grade engineering — not just a demo.</i>
</div>
