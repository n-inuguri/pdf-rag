#!/usr/bin/env python3
"""
Streamlit UI for retrieving from Chroma collections.

Run with: streamlit run src/app.py
"""
import os
import streamlit as st
from pathlib import Path

from dotenv import load_dotenv

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma

load_dotenv()

# Available collections
COLLECTIONS = [
    ("Sample", "sample.pdf"),
    ("Construction_Agreement", "Construction_Agreement.pdf"),
    ("Construction_Contract", "Construction_Contract-for-Major-Works.pdf")
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

def display_collection_info(db: Chroma, collection_name: str, persist_dir: str, model_name: str):
    """Display information about the selected collection."""
    try:
        count = db._collection.count()
        size = get_directory_size(persist_dir)
        st.subheader(f"Collection: {collection_name}")
        st.write(f"**Number of chunks:** {count}")
        st.write(f"**Database size:** {size} (shared across all collections)")
        st.write(f"**Embedding model:** {model_name}")
        st.write(f"**Persist directory:** {persist_dir}")
        if hasattr(db._collection, 'metadata') and db._collection.metadata:
            st.write(f"**Collection metadata:** {db._collection.metadata}")
    except Exception as e:
        st.error(f"Error retrieving collection info: {e}")

def main():
    st.title("Chroma Collection Query & Info")

    persist_dir = os.getenv("PERSIST_DIR", "../chroma_db")
    model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        top_k = int(os.getenv("TOP_K", "2"))
    except ValueError:
        top_k = 5

    persist_dir = str(Path(persist_dir).expanduser())
    if not Path(persist_dir).exists():
        st.error(f"Chroma persist directory not found: {persist_dir}")
        return

    emb = HuggingFaceEmbeddings(model_name=model_name)

    # Sidebar for selections
    st.sidebar.header("Options")
    action = st.sidebar.selectbox("Choose action", ["Retrieve", "Display Info"])
    collection_options = [f"{name} ({pdf})" for name, pdf in COLLECTIONS]
    selected_collection_display = st.sidebar.selectbox("Choose collection", collection_options)
    selected_collection_name = COLLECTIONS[collection_options.index(selected_collection_display)][0]

    # Load the collection
    db = Chroma(persist_directory=persist_dir, embedding_function=emb, collection_name=selected_collection_name)

    if action == "Retrieve":
        st.header("Retrieve from the Collection")
        query = st.text_input("Enter your query:")
        if st.button("Search"):
            if not query:
                st.warning("Please enter a query.")
            else:
                try:
                    results = db.similarity_search_with_score(query, k=top_k)
                except Exception:
                    docs = db.similarity_search(query, k=top_k)
                    results = [(d, None) for d in docs]

                for i, (doc, score) in enumerate(results, start=1):
                    st.subheader(f"Result #{i}")
                    if score is not None:
                        st.write(f"**Score:** {score}")
                    src = doc.metadata.get("source") if getattr(doc, "metadata", None) else None
                    if src:
                        st.write(f"**Source:** {src}")
                    text = doc.page_content.strip()
                    snippet = text if len(text) < 800 else text[:800] + "..."
                    st.write(snippet)

    elif action == "Display Info":
        st.header("Collection Information")
        display_collection_info(db, selected_collection_name, persist_dir, model_name)

if __name__ == "__main__":
    main()