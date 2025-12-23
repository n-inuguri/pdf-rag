#!/usr/bin/env python3
"""Small retrieval script for a named Chroma collection.

Reads configuration from `.env`:
- `PERSIST_DIR` (default: ../chroma_db)
- `MODEL_NAME` (default: sentence-transformers/all-MiniLM-L6-v2)
- `TOP_K` (optional, default: 5)

Interactive mode: Choose Retrieve, Display info, or Quit. Then select collection.

Usage:
  python src/retrieve_chroma.py
Then follow prompts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma

from utils import COLLECTIONS, perform_retrieve, _display_collection_info, display_results

load_dotenv()


def select_collection() -> str:
    """Prompt user to select a collection from available options."""
    print("Available collections:")
    for i, (col_name, pdf_name) in enumerate(COLLECTIONS, start=1):
        print(f"{i}. {col_name} ({pdf_name})")

    while True:
        try:
            choice = int(input(f"Select collection (1-{len(COLLECTIONS)}): "))
            if 1 <= choice <= len(COLLECTIONS):
                return COLLECTIONS[choice - 1][0]
            else:
                print(f"Invalid choice. Please select a number between 1 and {len(COLLECTIONS)}.")
        except ValueError:
            print("Please enter a number.")


def display_collection_info(db: Chroma, collection_name: str, persist_dir: str, model_name: str, document_name: str = None) -> None:
    """Display information about the selected collection."""
    _display_collection_info(db, collection_name, persist_dir, model_name, document_name, lambda x: print(f"\n{x}"), print)


def main() -> None:
    persist_dir = os.getenv("PERSIST_DIR", "../chroma_db")
    model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        top_k = int(os.getenv("TOP_K", "2"))
    except ValueError:
        top_k = 5

    persist_dir = str(Path(persist_dir).expanduser())
    if not Path(persist_dir).exists():
        raise SystemExit(f"Chroma persist directory not found: {persist_dir}")

    emb = HuggingFaceEmbeddings(model_name=model_name)

    while True:
        # Prompt for mode
        while True:
            print("Choose mode: \n 1. Retrieve \n 2. Display info \n 3. Quit ")
            mode = input("Choose mode: ").strip()
            if mode in ['1', '2', '3']:
                break
            else:
                print("Invalid choice. Please select 1, 2, or 3.")

        if mode == '3':
            print("Exiting.")
            break

        # Prompt for collection selection
        collection_name = select_collection()
        pdf_name = next(pdf for name, pdf in COLLECTIONS if name == collection_name)

        # Load the collection
        db = Chroma(persist_directory=persist_dir, embedding_function=emb, collection_name=collection_name)

        if mode == '1':
            # Retrieve mode
            # get query from argv or prompt
            if len(sys.argv) > 1:
                query = " ".join(sys.argv[1:])
            else:
                query = input("Enter your query: ")

            if not query:
                print("No query provided. Continuing.")
                continue

            results = perform_retrieve(db, query, top_k)
            display_results(results, lambda x: print(f"\n{x}"), print)
        else:
            # Display info mode
            display_collection_info(db, collection_name, persist_dir, model_name, pdf_name)

        # Ask to continue
        cont = input("\nDo you want to perform another action? (y/n): ").strip().lower()
        if cont != 'y':
            print("Exiting.")
            break


if __name__ == "__main__":
    main()
