#!/usr/bin/env python3
"""RAG chain using LangChain and Groq API for generation.

Reads configuration from `.env`:
- `PERSIST_DIR` (default: ../chroma_db)
- `MODEL_NAME` (default: sentence-transformers/all-MiniLM-L6-v2)
- `GROQ_API_KEY` (required)
- `GROQ_MODEL_NAME` (default: llama-3.1-8b-instant)
- `TOP_K` (optional, default: 5)

Prompts for collection selection, then runs RAG queries with Groq generation.

Usage:
  python src/rag_chain.py
Then follow prompts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def select_collection() -> str:
    """Prompt user to select a collection from available options."""
    collections = [
        ("Sample", "sample.pdf"),
        ("Construction_Agreement", "Construction_Agreement.pdf"),
        ("Construction_Contract", "Construction_Contract-for-Major-Works.pdf")
    ]
    print("Available collections:")
    for i, (col_name, pdf_name) in enumerate(collections, start=1):
        print(f"{i}. {col_name} ({pdf_name})")

    while True:
        try:
            choice = int(input("Select collection (1-3): "))
            if 1 <= choice <= len(collections):
                return collections[choice - 1][0]
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
        except ValueError:
            print("Please enter a number.")


def main() -> None:
    persist_dir = os.getenv("PERSIST_DIR", "../chroma_db")
    model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
    if not groq_api_key:
        raise SystemExit("GROQ_API_KEY not set in .env")
    try:
        top_k = int(os.getenv("TOP_K", "5"))
    except ValueError:
        top_k = 5

    persist_dir = str(Path(persist_dir).expanduser())
    if not Path(persist_dir).exists():
        raise SystemExit(f"Chroma persist directory not found: {persist_dir}")

    emb = HuggingFaceEmbeddings(model_name=model_name)

    # Prompt for collection selection
    collection_name = select_collection()

    # Load the collection
    db = Chroma(persist_directory=persist_dir, embedding_function=emb, collection_name=collection_name)

    # Set up RAG chain
    llm = ChatGroq(model=groq_model, api_key=groq_api_key)
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    print(f"\nRAG chain ready for collection '{collection_name}'. Type 'quit' to exit.")

    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() == 'quit':
            print("Exiting.")
            break
        if not query:
            print("Please enter a query.")
            continue

        try:
            result = qa_chain.invoke(query)
            print(f"\nAnswer: {result}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()