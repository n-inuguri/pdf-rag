# PDF-RAG-Framework: PDF → Chroma + Retrieval + RAG

This framework ingests PDF documents into a Chroma vector store (using HuggingFace embeddings), provides CLI and Streamlit UIs for retrieval, and supports Retrieval-Augmented Generation (RAG) answers via Groq.

Features:
- PDF ingestion with selectable loaders (PyMuPDF, PDFPlumber, PyPDF)
- Configurable chunking (size/overlap) with character-based splitter
- HuggingFace embeddings with configurable model
- Persistent Chroma vector store with per-collection counts
- Dynamic collection discovery in UI and CLI (from Chroma)
- Streamlit UI: List Collections, Upload PDF, Display info, Retrieve, RAG
- Unified CLI: List, Display, Retrieve, RAG, Quit (1–5) with runner script
- Retrieval via similarity search (optional scores) with source-aware snippets
- RAG answers via Groq (configurable model and API key)
- Source metadata stored per chunk for traceability
- Centralized configuration via `.env` and [src/config.py](src/config.py)
- Easy runs with `uv`; helper scripts [run_streamlit.sh](run_streamlit.sh) and [run_cli.sh](run_cli.sh)

This framework is suitable for any PDF-based document retrieval and RAG use case.
