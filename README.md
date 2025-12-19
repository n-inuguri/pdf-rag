# RAG ingestion: PDF -> Chroma (LangChain + HuggingFace embeddings)

This small project shows how to:

- Read a PDF and split it into token-based chunks (1024 tokens, 100 overlap).
- Create embeddings using a Hugging Face model.
- Persist embeddings to a Chroma vector store.

Files:

- [src/build_chroma.py](src/build_chroma.py) — main ingestion script.
- [requirements.txt](requirements.txt) — Python dependencies.

Quick start

1. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure `.env` and run the ingestion script (no CLI args required):

Edit the `.env` file in the project root and set `PDF_PATH`, `PERSIST_DIR`, `MODEL_NAME`, `CHUNK_SIZE`, and `CHUNK_OVERLAP`.

```bash
python src/build_chroma.py
```

Collections and multiple PDFs

- You can ingest multiple PDFs into the same Chroma persistence directory by giving each PDF a `COLLECTION_NAME` value in `.env`. If `COLLECTION_NAME` is blank, the script will default the collection name to the PDF filename (without extension).
- Each document chunk will have metadata `source` set to the PDF filename, so you can filter or display which PDF produced a result.

Querying a specific collection (example):

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=emb, collection_name="sample")
docs = db.similarity_search("your query", k=5)
```

Options:

- `--model_name` — Hugging Face model for embeddings (default: `sentence-transformers/all-MiniLM-L6-v2`).
- `--chunk_size` — tokens per chunk (default: 1024).
- `--chunk_overlap` — overlap tokens between chunks (default: 100).

Notes

- The script uses `TokenTextSplitter` with `encoding_name="cl100k_base"`. You can change that if you target a different tokenizer.
- Models will be downloaded from Hugging Face hub; ensure you have network access.
