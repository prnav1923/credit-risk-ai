import os
import json
import logging
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    ContextPrecisionWithReference,
    ContextRecall,
    SemanticSimilarity,
    NonLLMStringSimilarity
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langsmith import Client
import mlflow

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Test dataset ---
EVAL_QUESTIONS = [
    {
        "question": "What is the minimum FICO score required for loan approval?",
        "ground_truth": "The minimum FICO score required for loan approval is 650."
    },
    {
        "question": "What DTI ratio triggers automatic decline?",
        "ground_truth": "A DTI ratio above 50% triggers automatic decline."
    },
    {
        "question": "What are the high risk indicators?",
        "ground_truth": "High risk indicators include revolving utilization above 75%, public records or bankruptcies, multiple hard inquiries in last 6 months, delinquency history in last 2 years, and loan to income ratio above 0.4."
    },
    {
        "question": "What is the interest rate for Grade A loans?",
        "ground_truth": "Grade A loans have interest rates between 5.99% and 8.99%."
    },
    {
        "question": "When is a loan charged off?",
        "ground_truth": "A loan is charged off when it is 90 days past due and sent to a collections agency."
    },
    {
        "question": "What triggers a mandatory model review?",
        "ground_truth": "An AUC-ROC below 0.70 triggers a mandatory model review."
    },
    {
        "question": "What are the manual review criteria?",
        "ground_truth": "Manual review criteria include risk score between 0.3 and 0.6, income not verified, loan purpose of small business or medical, first time borrower with limited credit history, and joint applications with co-borrower DTI concerns."
    }
]


# --- Load vectorstore ---
def get_vectorstore():
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )


# --- Generate answers using RAG ---
def generate_rag_answers(vectorstore):
    logger.info("Generating RAG answers for evaluation dataset...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in EVAL_QUESTIONS:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # Retrieve context
        docs = vectorstore.similarity_search(question, k=3)
        context = [doc.page_content for doc in docs]

        # Simple answer extraction from context
        # In production this would use the full RAG chain
        combined_context = "\n".join(context)

        # Use context as answer proxy for evaluation
        answer = combined_context[:500]

        questions.append(question)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(ground_truth)

        logger.info(f"Q: {question[:50]}... → {len(docs)} chunks retrieved")

    return questions, answers, contexts, ground_truths


# --- Run Ragas evaluation ---
def run_ragas_evaluation(questions, answers, contexts, ground_truths):
    logger.info("Running evaluation with sentence-transformers...")

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    model = SentenceTransformer(EMBEDDING_MODEL)

    semantic_scores = []
    context_recall_scores = []

    for i, (answer, ground_truth, context_list) in enumerate(
        zip(answers, ground_truths, contexts)
    ):
        # Semantic similarity — answer vs ground truth
        answer_emb = model.encode([answer[:500]])
        gt_emb = model.encode([ground_truth])
        sem_score = float(cosine_similarity(answer_emb, gt_emb)[0][0])
        semantic_scores.append(sem_score)

        # Context recall — is ground truth covered by retrieved context?
        combined_context = " ".join(context_list)
        context_emb = model.encode([combined_context[:500]])
        recall_score = float(cosine_similarity(gt_emb, context_emb)[0][0])
        context_recall_scores.append(recall_score)

        logger.info(
            f"Q{i+1}: semantic={sem_score:.3f} | context_recall={recall_score:.3f}"
        )

    results = {
        "semantic_similarity": float(np.mean(semantic_scores)),
        "context_recall": float(np.mean(context_recall_scores)),
    }

    logger.info("Evaluation complete ✅")
    return results


# --- Log to MLflow ---
def log_to_mlflow(results: dict):
    logger.info("Logging evaluation results to MLflow...")
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("credit-risk-rag-evaluation")

    with mlflow.start_run(run_name="rag-evaluation"):
        for metric, value in results.items():
            mlflow.log_metric(metric, value)
            logger.info(f"{metric}: {value:.4f}")

    logger.info("Logged to MLflow ✅")


# --- Log to LangSmith ---
def log_to_langsmith(questions, answers, contexts, results):
    logger.info("Logging to LangSmith...")
    try:
        client = Client()

        # Create dataset in LangSmith
        dataset_name = "credit-risk-rag-eval"
        try:
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="Credit risk RAG evaluation dataset"
            )
        except Exception:
            dataset = client.read_dataset(dataset_name=dataset_name)

        # Add examples
        for i, (q, a) in enumerate(zip(questions, answers)):
            client.create_example(
                inputs={"question": q},
                outputs={"answer": a},
                dataset_id=dataset.id
            )

        logger.info(f"Logged {len(questions)} examples to LangSmith ✅")

    except Exception as e:
        logger.warning(f"LangSmith logging failed: {e}")


# --- Main evaluation pipeline ---
def run_evaluation():
    logger.info("=== Starting RAG Evaluation Pipeline ===")

    # Load vectorstore
    vectorstore = get_vectorstore()

    # Generate answers
    questions, answers, contexts, ground_truths = generate_rag_answers(vectorstore)

    # Run Ragas
    results = run_ragas_evaluation(questions, answers, contexts, ground_truths)

    # Print results
    print("\n=== RAGAS EVALUATION RESULTS ===")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric:25s}: {value:.4f}")
    print("================================\n")

    # Log to MLflow
    log_to_mlflow(dict(results))

    # Log to LangSmith
    log_to_langsmith(questions, answers, contexts, dict(results))

    logger.info("=== Evaluation Complete ✅ ===")
    return results


if __name__ == "__main__":
    run_evaluation()