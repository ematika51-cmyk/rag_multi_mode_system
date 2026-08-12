# RAG Document Processing & Stylometry Pipeline

A Retrieval-Augmented Generation (RAG) pipeline built with LangChain and Python. The project ingests documents (PDF and DOCX), splits them into manageable text chunks, and evaluates writing style metrics using stylometric analysis and style geometry (embedding cosine similarity).

## Features
- Direct PDF and DOCX document ingestion
- ChromaDB vector store integration
- Dual-mode processing (Standard QA & Style Mimicking)

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt