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

from utils import COLLECTIONS, perform_retrieve, _display_collection_info, display_results, setup_rag_chain

load_dotenv()

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def display_collection_info(db: Chroma, collection_name: str, persist_dir: str, model_name: str, document_name: str = None):
    """Display information about the selected collection."""
    _display_collection_info(db, collection_name, persist_dir, model_name, document_name, st.subheader, st.write)

def main():
    st.title("Contract Documents Analysis")

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
    action = st.sidebar.selectbox("Choose action", ["Display", "Retrieve", "RAG"])
    collection_options = [f"{name} ({pdf})" for name, pdf in COLLECTIONS]
    selected_collection_display = st.sidebar.selectbox("Choose collection", collection_options)
    selected_collection_name = COLLECTIONS[collection_options.index(selected_collection_display)][0]
    pdf_name = COLLECTIONS[collection_options.index(selected_collection_display)][1]

    # Load the collection
    db = Chroma(persist_directory=persist_dir, embedding_function=emb, collection_name=selected_collection_name)

    if action == "Retrieve":
        st.header("Retrieve from the Document")
        query = st.text_input("Enter your query:")
        if st.button("Search"):
            if not query:
                st.warning("Please enter a query.")
            else:
                results = perform_retrieve(db, query, top_k)
                display_results(results, st.subheader, st.write)

    elif action == "Display":
        st.header("Document Information")
        display_collection_info(db, selected_collection_name, persist_dir, model_name, pdf_name)

    elif action == "RAG":
        st.header("RAG Query with Groq")
        groq_api_key = os.getenv("GROQ_API_KEY")
        groq_model = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
        if not groq_api_key:
            st.error("GROQ_API_KEY not set in .env")
        else:
            try:
                top_k = int(os.getenv("TOP_K", "5"))
            except ValueError:
                top_k = 5

            # Set up RAG chain
            qa_chain = setup_rag_chain(db, groq_api_key, groq_model, top_k)

            # Show which collection is in use for RAG
            st.caption(f"Collection: {selected_collection_name} ({pdf_name})")
            query = st.text_input("Enter your query for RAG:")
            if st.button("Generate Answer"):
                if not query:
                    st.warning("Please enter a query.")
                else:
                    with st.spinner("Generating answer..."):
                        try:
                            result = qa_chain.invoke(query)
                            st.markdown("**Answer:**")
                            st.write(result['result'])
                        except Exception as e:
                            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()