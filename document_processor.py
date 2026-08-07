import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_document(file_path):
    # Step 1: Select the Loader Based on the File Extension
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx') or file_path.endswith('.doc'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError("Unsupported file format! Please use PDF, DOCX, or TXT.")

    print(f"Loading document: {file_path}...")
    documents = loader.load()
    print(f"Total Pages/Sections Loaded: {len(documents)}")

    # Step 2: Configure Text Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Maximum ~500 characters per chunk
        chunk_overlap=50      # Keep a 50-character overlap to preserve context
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Total Chunks Created: {len(chunks)}")
    return chunks

# --- Main Execution ---

if __name__ == "__main__":
    # Exact name of your file
    sample_file = "sample.docx"

    if os.path.exists(sample_file):
        chunks = load_and_split_document(sample_file)

        # Check whether chunks were created successfully
        if len(chunks) > 0:
            print("\n--- Sample Chunk 1 ---")
            print(chunks[0].page_content)
        else:
            print("\n⚠️ Warning: Unable to extract text from the document! Please check that the file contains readable content.")
    else:
        print(f"❌ Error: '{sample_file}' was not found! Make sure the file is saved inside your project folder (rag_project).")