import os
import logging
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --- Load vectorstore ---
def get_vectorstore():
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    return vectorstore


# --- Format retrieved docs ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# --- Build RAG chain ---
def get_rag_chain():
    logger.info("Building RAG chain...")

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    prompt = PromptTemplate.from_template("""Policy expert. Answer briefly using context only.
If not in context, say "Not covered in policy."

Context: {context}

Q: {question}
A:""")

    # Load lightweight local LLM
    logger.info("Loading LLM...")
    hf_pipeline = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=128,
        truncation=True,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    llm = HuggingFacePipeline(
        pipeline=hf_pipeline,
        pipeline_kwargs={"truncation": True}
    )

    # LCEL chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("RAG chain ready ✅")
    return rag_chain, retriever


# --- Query function ---
def query_policy(question: str, rag_chain=None, retriever=None) -> dict:
    if rag_chain is None:
        rag_chain, retriever = get_rag_chain()

    logger.info(f"Query: {question}")
    answer = rag_chain.invoke(question)
    source_docs = retriever.invoke(question)

    return {
        "question": question,
        "answer": answer,
        "source_chunks": len(source_docs)
    }


if __name__ == "__main__":
    chain, retriever = get_rag_chain()

    questions = [
        "What is the minimum FICO score required for loan approval?",
        "What DTI ratio triggers automatic decline?",
        "What are the high risk indicators?"
    ]

    for q in questions:
        result = query_policy(q, chain, retriever)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['source_chunks']} chunks")