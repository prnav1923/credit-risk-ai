import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
POLICY_PATH = "data/policies/credit_risk_policy.txt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    logger.info(f"Loading documents from {POLICY_PATH}")
    loader = TextLoader(POLICY_PATH)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} document(s)")
    return documents


def split_documents(documents):
    logger.info("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def get_embeddings():
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings


def ingest_to_pinecone(chunks, embeddings):
    logger.info(f"Ingesting to Pinecone index: {PINECONE_INDEX_NAME}")
    
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )

    logger.info(f"Ingested {len(chunks)} chunks to Pinecone ✅")
    return vectorstore


def run_ingestion():
    logger.info("=== Starting RAG Ingestion ===")
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    vectorstore = ingest_to_pinecone(chunks, embeddings)
    logger.info("=== Ingestion Complete ✅ ===")
    return vectorstore


if __name__ == "__main__":
    run_ingestion()