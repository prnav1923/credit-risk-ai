import sys
import os

# Add project root to path so src imports work
sys.path.insert(0, '/Users/pranav/Code/credit-risk-ai')

import asyncio
import json
import logging
import os
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# --- Config ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

# Sample application for testing
SAMPLE_APPLICATION = {
    "loan_amnt": 10000,
    "int_rate": 12.5,
    "installment": 350.0,
    "annual_inc": 60000,
    "dti": 18.5,
    "delinq_2yrs": 0,
    "fico_range_low": 680,
    "fico_range_high": 684,
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
    "avg_cur_bal": 8000,
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "grade": "B",
    "sub_grade": "B3",
    "initial_list_status": "w",
    "application_type": "Individual"
}

# --- MCP Server ---
server = Server("credit-risk-ai")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="assess_credit_risk",
            description=(
                "Assess credit risk for a loan application. "
                "Returns risk score (0-1), decision (APPROVE/REVIEW/DECLINE), "
                "and risk level (LOW/MEDIUM/HIGH). "
                "Use this first for any loan application assessment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "application": {
                        "type": "object",
                        "description": "Loan application data. If not provided, uses sample data.",
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="explain_credit_decision",
            description=(
                "Explain what factors drove the credit risk decision using SHAP values. "
                "Returns top risk factors and protective factors with impact scores. "
                "Use after assess_credit_risk to understand the decision."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "application": {
                        "type": "object",
                        "description": "Loan application data. If not provided, uses sample data.",
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="query_credit_policy",
            description=(
                "Query the credit risk policy knowledge base. "
                "Ask questions about lending criteria, risk grades, "
                "approval thresholds, decline criteria, and compliance requirements."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Policy question to answer. Example: 'What is the minimum FICO score for approval?'"
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="run_full_agent_analysis",
            description=(
                "Run a complete AI agent analysis of a loan application. "
                "The agent uses Groq Llama 3.3 70B to reason through "
                "risk assessment, policy compliance, and similar cases "
                "before making a final recommendation. "
                "Most comprehensive analysis available."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "application": {
                        "type": "object",
                        "description": "Loan application data. If not provided, uses sample data.",
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_model_info",
            description=(
                "Get information about the credit risk model — "
                "AUC-ROC score, training data size, features used, "
                "and system version."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
    name="override_credit_decision",
    description=(
        "Override the AI credit risk decision with a human underwriter decision. "
        "Use this when a human reviewer disagrees with the model's recommendation. "
        "Logs the override to S3 audit trail for compliance. "
        "Required: application data, override decision (APPROVE/REVIEW/DECLINE), "
        "reviewer ID, and reason for override."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "application": {
                "type": "object",
                "description": "Loan application data.",
            },
            "override_decision": {
                "type": "string",
                "enum": ["APPROVE", "REVIEW", "DECLINE"],
                "description": "Human reviewer's decision."
            },
            "reviewer_id": {
                "type": "string",
                "description": "ID of the human reviewer e.g. underwriter_001"
            },
            "reason": {
                "type": "string",
                "description": "Reason for overriding the AI decision."
            }
        },
        "required": ["override_decision", "reviewer_id", "reason"]
    }
),
        Tool(
            name="get_audit_log",
            description=(
                "Retrieve the audit log of all human override decisions. "
                "Shows reviewer ID, original AI decision, override decision, "
                "reason, and timestamp for each override. "
                "Use this to review compliance history."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),  
        Tool(
    name="assess_combined",
    description=(
        "Run combined credit risk AND fraud detection assessment. "
        "Returns credit risk score, fraud score, fraud indicators, "
        "and combined decision in one call. "
        "Use this for a complete risk picture of any loan application."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "application": {
                "type": "object",
                "description": "Loan application data. If not provided uses sample data."
            }
        },
        "required": []
    }
),
Tool(
    name="run_multi_agent_analysis",
    description=(
        "Run full 3-agent workflow: Risk Agent → Compliance Agent → Decision Agent. "
        "Most comprehensive analysis available. "
        "Risk Agent assesses credit + fraud. "
        "Compliance Agent checks ECOA/FCRA policy. "
        "Decision Agent synthesizes final recommendation. "
        "Takes ~60 seconds but returns the most thorough analysis."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "application": {
                "type": "object",
                "description": "Loan application data. If not provided uses sample data."
            }
        },
        "required": []
    }
),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    async with httpx.AsyncClient(timeout=120.0) as client:

        # --- Tool 1: Assess Credit Risk ---
        if name == "assess_credit_risk":
            application = arguments.get("application", SAMPLE_APPLICATION)
            try:
                async with httpx.AsyncClient(timeout=120.0) as inner_client:
                    response = await client.post(
                        f"{API_BASE_URL}/v1/predict",
                        headers=HEADERS,
                        json=application
                    )
                result = response.json()
                output = (
                    f"Credit Risk Assessment:\n"
                    f"Risk Score: {result['risk_score']} (0=Safe, 1=Default)\n"
                    f"Decision: {result['decision']}\n"
                    f"Risk Level: {result['risk_level']}\n"
                    f"Confidence: {result['confidence']}\n"
                    f"Message: {result['message']}"
                )
                return [TextContent(type="text", text=output)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]

        # --- Tool 2: Explain Decision ---
        elif name == "explain_credit_decision":
            application = arguments.get("application", SAMPLE_APPLICATION)
            try:
                response = await client.post(
                    f"{API_BASE_URL}/v1/explain",
                    headers=HEADERS,
                    json=application
                )
                result = response.json()
                risk_factors = "\n".join([
                    f"  - {f['feature']}: {f['impact']}"
                    for f in result['top_risk_factors']
                ])
                protective_factors = "\n".join([
                    f"  - {f['feature']}: {f['impact']}"
                    for f in result['top_protective_factors']
                ])
                output = (
                    f"Decision Explanation (SHAP Values):\n"
                    f"Risk Score: {result['risk_score']}\n"
                    f"Decision: {result['decision']}\n\n"
                    f"Top Risk Factors (increasing default probability):\n"
                    f"{risk_factors}\n\n"
                    f"Top Protective Factors (decreasing default probability):\n"
                    f"{protective_factors}"
                )
                return [TextContent(type="text", text=output)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]

        # --- Tool 3: Query Policy ---
        elif name == "query_credit_policy":
            question = arguments.get("question", "What are the loan approval criteria?")
            try:
                from src.rag.retriever import get_vectorstore
                vectorstore = get_vectorstore()
                docs = vectorstore.similarity_search(question, k=3)
                context = "\n\n".join(doc.page_content for doc in docs)
                output = (
                    f"Policy Query: {question}\n\n"
                    f"Relevant Policy:\n{context}"
                )
                return [TextContent(type="text", text=output)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]

        # --- Tool 4: Run Full Agent Analysis ---
        elif name == "run_full_agent_analysis":
            application = arguments.get("application", SAMPLE_APPLICATION)
            try:
                response = await client.post(
                    f"{API_BASE_URL}/v1/agent",
                    headers=HEADERS,
                    json={"application": application},
                    timeout=120.0
                )
                result = response.json()
                return [TextContent(type="text", text=result['report'])]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]

        # --- Tool 5: Get Model Info ---
        elif name == "get_model_info":
            try:
                response = await client.get(
                    f"{API_BASE_URL}/v1/model-info",
                    headers=HEADERS
                )
                result = response.json()
                output = (
                    f"Credit Risk Model Information:\n"
                    f"Model Type: {result['model_type']}\n"
                    f"AUC-ROC: {result['auc_roc']}\n"
                    f"PR-AUC: {result['pr_auc']}\n"
                    f"Features: {result['features']}\n"
                    f"Training Samples: {result['training_samples']:,}\n"
                    f"Test Samples: {result['test_samples']:,}\n"
                    f"Default Rate: {result['default_rate']}\n"
                    f"Dataset: {result['dataset']}\n"
                    f"Version: {result['version']}"
                )
                return [TextContent(type="text", text=output)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]
        
        # --- Tool 6: Override Decision ---
        elif name == "override_credit_decision":
            application = arguments.get("application", SAMPLE_APPLICATION)
            override_decision = arguments.get("override_decision")
            reviewer_id = arguments.get("reviewer_id")
            reason = arguments.get("reason")
            try:
                response = await client.post(
                    f"{API_BASE_URL}/v1/override",
                    headers=HEADERS,
                    json={
                        "application": application,
                        "override_decision": override_decision,
                        "reviewer_id": reviewer_id,
                        "reason": reason
                    }
                )
                result = response.json()
                output = (
                    f"✅ Override Recorded Successfully\n\n"
                    f"Original AI Decision: {result['original_decision']}\n"
                    f"Human Override: {result['override_decision']}\n"
                    f"Reviewer: {result['reviewer_id']}\n"
                    f"Reason: {result['reason']}\n"
                    f"Timestamp: {result['timestamp']}\n"
                    f"Status: Logged to S3 audit trail for compliance"
                )
                return [TextContent(type="text", text=output)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]
            
            # --- Tool 7: Get Audit Log ---
        elif name == "get_audit_log":
            try:
                response = await client.get(
                    f"{API_BASE_URL}/v1/audit",
                    headers=HEADERS
                )
                result = response.json()
                
                if result['total_overrides'] == 0:
                    return [TextContent(type="text", text="No overrides recorded yet.")]
                
                output = f"Audit Log — {result['total_overrides']} override(s) recorded:\n\n"
            
                for i, override in enumerate(result['overrides'], 1):
                    output += (
                        f"Override #{i}\n"
                        f"  Timestamp:         {override['timestamp']}\n"
                        f"  Reviewer:          {override['reviewer_id']}\n"
                        f"  Original Decision: {override['original_decision']}\n"
                        f"  Override Decision: {override['override_decision']}\n"
                        f"  Risk Score:        {override['risk_score']}\n"
                        f"  Reason:            {override['reason']}\n"
                        f"  ─────────────────────────────────\n"
                    )
        
                output += "\nAll overrides logged to S3 for regulatory compliance."
                return [TextContent(type="text", text=output)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]

        # --- Tool 8: Combined Assessment ---
        elif name == "assess_combined":
            application = arguments.get("application", SAMPLE_APPLICATION)
            try:
                async with httpx.AsyncClient(timeout=120.0) as inner_client:
                    response = await inner_client.post(
                        f"{API_BASE_URL}/v1/assess",
                        headers=HEADERS,
                        json=application,
                        timeout=60.0
                    )
        
                # Debug — show raw response if parsing fails
                raw = response.json()
        
                # Check if error returned
                if "detail" in raw:
                    return [TextContent(type="text", text=f"API Error: {raw['detail']}")]
        
                fraud_indicators = "\n".join(
                    [f"  - {i}" for i in raw.get("fraud_indicators", [])]
                ) or "  None detected"

                output = (
                    f"Combined Credit Risk + Fraud Assessment:\n\n"
                    f"CREDIT RISK:\n"
                    f"  Score: {raw.get('credit_risk_score', 'N/A')}\n"
                    f"  Decision: {raw.get('credit_decision', 'N/A')}\n"
                    f"  Risk Level: {raw.get('credit_risk_level', 'N/A')}\n\n"
                    f"FRAUD DETECTION:\n"
                    f"  Score: {raw.get('fraud_score', 'N/A')}\n"
                    f"  Flag: {'⚠️ FRAUD DETECTED' if raw.get('fraud_flag') else '✅ Clean'}\n"
                    f"  Risk: {raw.get('fraud_risk', 'N/A')}\n"
                    f"  Indicators:\n{fraud_indicators}\n\n"
                    f"COMBINED DECISION: {raw.get('combined_decision', 'N/A')}\n"
                    f"COMBINED RISK: {raw.get('combined_risk', 'N/A')}\n"
                    f"MESSAGE: {raw.get('message', 'N/A')}"
                )
                return [TextContent(type="text", text=output)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error details: {str(e)}\nRaw response: {response.text if 'response' in dir() else 'No response'}")]
        # --- Tool 9: Multi-Agent Analysis ---
        elif name == "run_multi_agent_analysis":
            application = arguments.get("application", SAMPLE_APPLICATION)
            try:
                response = await client.post(
                    f"{API_BASE_URL}/v1/multi-agent",
                    headers=HEADERS,
                    json={"application": application},
                    timeout=180.0
                )
                result = response.json()
                return [TextContent(type="text", text=result['report'])]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]


# --- Run server ---
async def main():
    logger.info("Starting Credit Risk AI MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())