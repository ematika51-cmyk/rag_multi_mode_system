# RAG Document Processing & Stylometry Pipeline

A Retrieval-Augmented Generation (RAG) pipeline built with LangChain and Python. The project ingests documents (PDF and DOCX), splits them into manageable text chunks, and evaluates writing style metrics using stylometric analysis and style geometry (embedding cosine similarity).

## Features

- **Multi-Format Ingestion**: Supports `.docx` and `.pdf` files.
- **Text Chunking**: Uses `RecursiveCharacterTextSplitter` with configurable chunk sizes and overlaps.
- **Stylometry Analysis**: Calculates word counts, average sentence length, and vocabulary richness.
- **Style Geometry**: Computes cosine similarity scores using HuggingFace embeddings (`all-MiniLM-L6-v2`).

## Project Structure

```text
rag_project/
│
├── my_files/          # Ingested reference documents (.docx, .pdf)
├── app.py             # Main execution script
├── .gitignore         # Environment and cache ignore rules
└── README.md          # Project documentation
