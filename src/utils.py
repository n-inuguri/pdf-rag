"""
Common utilities for RAG project.
"""
import os

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq

# Available collections
COLLECTIONS = [
    ("Construction_Agreement", "Construction_Agreement.pdf"),
    ("Construction_Contract", "Construction_Contract-for-Major-Works.pdf"),
    ("Construction_Contract2", "Construction_Contract-for-Major-Works.pdf"),
]

def get_directory_size(path: str) -> str:
    """Calculate the total size of a directory in KB or MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except OSError:
                pass
    if total_size < 1024 * 1024:
        return f"{total_size / 1024:.2f} KB"
    else:
        return f"{total_size / (1024 * 1024):.2f} MB"

def perform_retrieve(db, query: str, top_k: int) -> list:
    """Perform retrieval and return list of (doc, score) tuples."""
    try:
        results = db.similarity_search_with_score(query, k=top_k)
    except Exception:
        docs = db.similarity_search(query, k=top_k)
        results = [(d, None) for d in docs]
    return results

def _display_collection_info(db, collection_name: str, persist_dir: str, model_name: str, document_name: str = None, subheader_func=None, write_func=None):
    """Generic display collection info."""
    try:
        count = db._collection.count()
        size = get_directory_size(persist_dir)
        subheader_func(f"Collection: {collection_name}")
        if document_name:
            write_func(f"Document: {document_name}")
        write_func(f"Number of chunks: {count}")
        write_func(f"Database size: {size} (shared across all collections)")
        write_func(f"Embedding model: {model_name}")
        write_func(f"Persist directory: {persist_dir}")
        if hasattr(db._collection, 'metadata') and db._collection.metadata:
            write_func(f"Collection metadata: {db._collection.metadata}")
    except Exception as e:
        write_func(f"Error retrieving collection info: {e}")

def display_results(results: list, subheader_func, write_func):
    """Generic display results."""
    for i, (doc, score) in enumerate(results, start=1):
        subheader_func(f"Result #{i}")
        if score is not None:
            write_func(f"Score: {score}")
        src = doc.metadata.get("source") if getattr(doc, "metadata", None) else None
        if src:
            write_func(f"Source: {src}")
        text = doc.page_content.strip()
        snippet = text if len(text) < 800 else text[:800] + "..."
        write_func(snippet)


def setup_rag_chain(db, groq_api_key: str, groq_model: str, top_k: int):
    """Set up RAG chain with Groq LLM."""
    llm = ChatGroq(model=groq_model, api_key=groq_api_key)
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain