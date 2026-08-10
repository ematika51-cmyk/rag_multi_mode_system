import os
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_assignments(folder_path="./my_assignments"):
    # 1. PDFs aur Word files ko folder se load karna
    pdf_loader = PyPDFDirectoryLoader(folder_path)
    docx_loader = DirectoryLoader(folder_path, glob="**/*.docx", loader_cls=Docx2txtLoader)
    
    docs = pdf_loader.load() + docx_loader.load()
    
    # 2. Text Chunks banana
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )
    return text_splitter.split_documents(docs)