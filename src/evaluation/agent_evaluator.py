import os
import json
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import mlflow

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# --- Judge LLM ---
def get_judge_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

# --- Evaluation Dataset ---
EVAL_CASES = [
    {
        "name": "Clear Approve — Grade A",
        "application": {
            "loan_amnt": 5000, "int_rate": 6.5, "installment": 150.0,
            "annual_inc": 120000, "dti": 8.0, "delinq_2yrs": 0,
            "fico_range_low": 760, "fico_range_high": 764,
            "open_acc": 8, "pub_rec": 0, "revol_bal": 5000,
            "revol_util": 10.0, "total_acc": 20, "emp_length": 8,
            "mort_acc": 1, "pub_rec_bankruptcies": 0,
            "num_actv_bc_tl": 3, "bc_util": 15.0,
            "percent_bc_gt_75": 0.0, "avg_cur_bal": 10000,
            "home_ownership": "MORTGAGE", "verification_status": "Verified",
            "purpose": "debt_consolidation", "grade": "A",
            "sub_grade": "A2", "initial_list_status": "w",
            "application_type": "Individual"
        },
        "expected_decision": "APPROVE",
        "policy_check": "Grade A borrower, FICO 760+, DTI 8% — should approve per policy Section 2"
    },
    {
        "name": "Clear Decline — Grade F with Fraud",
        "application": {
            "loan_amnt": 35000, "int_rate": 28.99, "installment": 980.0,
            "annual_inc": 32000, "dti": 42.0, "delinq_2yrs": 3,
            "fico_range_low": 605, "fico_range_high": 609,
            "open_acc": 18, "pub_rec": 2, "revol_bal": 28000,
            "revol_util": 88.0, "total_acc": 30, "emp_length": 1,
            "mort_acc": 0, "pub_rec_bankruptcies": 1,
            "num_actv_bc_tl": 8, "bc_util": 90.0,
            "percent_bc_gt_75": 80.0, "avg_cur_bal": 3000,
            "home_ownership": "RENT", "verification_status": "Not Verified",
            "purpose": "small_business", "grade": "F",
            "sub_grade": "F3", "initial_list_status": "f",
            "application_type": "Individual"
        },
        "expected_decision": "DECLINE",
        "policy_check": "Grade F, bankruptcy, DTI 42%, not verified — must decline per policy Section 3"
    },
    {
        "name": "Review Case — Borderline",
        "application": {
            "loan_amnt": 10000, "int_rate": 12.5, "installment": 350.0,
            "annual_inc": 60000, "dti": 18.5, "delinq_2yrs": 0,
            "fico_range_low": 680, "fico_range_high": 684,
            "open_acc": 10, "pub_rec": 0, "revol_bal": 15000,
            "revol_util": 45.0, "total_acc": 25, "emp_length": 5,
            "mort_acc": 2, "pub_rec_bankruptcies": 0,
            "num_actv_bc_tl": 4, "bc_util": 50.0,
            "percent_bc_gt_75": 25.0, "avg_cur_bal": 8000,
            "home_ownership": "RENT", "verification_status": "Verified",
            "purpose": "debt_consolidation", "grade": "B",
            "sub_grade": "B3", "initial_list_status": "w",
            "application_type": "Individual"
        },
        "expected_decision": "REVIEW",
        "policy_check": "Score 0.3-0.6 band — manual review required per policy Section 4"
    }
]


def judge_agent_response(
    agent_report: str,
    expected_decision: str,
    policy_check: str,
    case_name: str
) -> dict:
    """
    Use LLM-as-judge to evaluate agent response quality.
    Scores on 4 dimensions:
    1. Decision accuracy — did agent make the right call?
    2. Policy adherence — did agent cite correct policy?
    3. Reasoning quality — is the reasoning sound?
    4. Regulatory compliance — ECOA/FCRA mentioned?
    """
    judge = get_judge_llm()

    judge_prompt = f"""You are an expert credit risk auditor evaluating an AI agent's credit decision.

CASE: {case_name}
EXPECTED DECISION: {expected_decision}
POLICY CHECK: {policy_check}

AGENT REPORT:
{agent_report}

Evaluate the agent's response on these 4 dimensions.
Return ONLY valid JSON, no other text:

{{
    "decision_accuracy": {{
        "score": <0.0-1.0>,
        "reasoning": "<did agent make correct decision?>"
    }},
    "policy_adherence": {{
        "score": <0.0-1.0>,
        "reasoning": "<did agent cite relevant policy sections?>"
    }},
    "reasoning_quality": {{
        "score": <0.0-1.0>,
        "reasoning": "<is the reasoning sound and justified?>"
    }},
    "regulatory_compliance": {{
        "score": <0.0-1.0>,
        "reasoning": "<did agent mention ECOA/FCRA requirements?>"
    }},
    "overall_score": <0.0-1.0>,
    "pass": <true/false>,
    "feedback": "<one sentence summary>"
}}"""

    response = judge.invoke([HumanMessage(content=judge_prompt)])

    try:
        # Clean response and parse JSON
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        result = json.loads(content)
        result["case_name"] = case_name
        result["expected_decision"] = expected_decision
        return result
    except Exception as e:
        logger.error(f"Failed to parse judge response: {e}")
        return {
            "case_name": case_name,
            "error": str(e),
            "overall_score": 0,
            "pass": False
        }


def run_agent_for_evaluation(application: dict) -> str:
    """Run the multi-agent workflow for evaluation"""
    import sys
    sys.path.insert(0, '/Users/pranav/Code/credit-risk-ai')
    from src.agent.multi_agent import run_multi_agent
    return run_multi_agent(application)


def run_agent_evaluation():
    """
    Full LLM-as-judge evaluation pipeline.
    Runs 3 test cases through multi-agent and judges responses.
    """
    logger.info("=== Starting LLM-as-Judge Agent Evaluation ===")

    results = []
    total_score = 0

    for i, case in enumerate(EVAL_CASES):
        logger.info(f"Evaluating case {i+1}/{len(EVAL_CASES)}: {case['name']}")

        # Run agent
        logger.info("Running multi-agent workflow...")
        agent_report = run_agent_for_evaluation(case["application"])

        # Judge the response
        logger.info("Running LLM judge...")
        judgment = judge_agent_response(
            agent_report=agent_report,
            expected_decision=case["expected_decision"],
            policy_check=case["policy_check"],
            case_name=case["name"]
        )

        judgment["agent_report_preview"] = agent_report[:300] + "..."
        results.append(judgment)

        score = judgment.get("overall_score", 0)
        total_score += score

        logger.info(f"Case {i+1} score: {score:.2f} | Pass: {judgment.get('pass', False)}")

    avg_score = total_score / len(EVAL_CASES)
    passed = sum(1 for r in results if r.get("pass", False))

    # Print results
    print("\n=== LLM-AS-JUDGE EVALUATION RESULTS ===")
    for r in results:
        print(f"\nCase: {r['case_name']}")
        print(f"Overall Score: {r.get('overall_score', 0):.2f}")
        print(f"Pass: {'✅' if r.get('pass') else '❌'}")
        if 'feedback' in r:
            print(f"Feedback: {r['feedback']}")
        if 'decision_accuracy' in r:
            print(f"Decision Accuracy: {r['decision_accuracy']['score']:.2f}")
            print(f"Policy Adherence: {r['policy_adherence']['score']:.2f}")
            print(f"Reasoning Quality: {r['reasoning_quality']['score']:.2f}")
            print(f"Regulatory Compliance: {r['regulatory_compliance']['score']:.2f}")

    print(f"\n{'='*40}")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Passed: {passed}/{len(EVAL_CASES)}")
    print(f"{'='*40}\n")

    # Log to MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("agent-evaluation")

    with mlflow.start_run(run_name="llm-as-judge"):
        mlflow.log_metric("avg_score", avg_score)
        mlflow.log_metric("cases_passed", passed)
        mlflow.log_metric("total_cases", len(EVAL_CASES))

        for r in results:
            name = r['case_name'].replace(' ', '_').replace('—', '').strip()
            mlflow.log_metric(f"{name}_score", r.get('overall_score', 0))

        mlflow.log_dict({"results": results}, "evaluation_results.json")
        logger.info("Results logged to MLflow ✅")

    logger.info("=== Agent Evaluation Complete ✅ ===")
    return results, avg_score


if __name__ == "__main__":
    results, avg_score = run_agent_evaluation()