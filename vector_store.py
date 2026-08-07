import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from document_processor import load_and_split_document

def create_and_save_vector_db(file_path, db_directory="./chroma_db"):
    # 1. Load and Chunk the Document (Phase 2 Logic)
    print("Step 1: Starting document processing...")
    chunks = load_and_split_document(file_path)

    if not chunks:
        print("❌ Chunks are empty. Vector Database cannot be created.")
        return None

    # 2. Initialize the Free & Local Embedding Model
    print("\nStep 2: Loading Embedding Model (MiniLM)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Convert Chunks into Embeddings and Save Them in ChromaDB
    print("\nStep 3: Saving Text Chunks into the Vector Store (ChromaDB)...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_directory
    )

    print(f"\n✅ SUCCESS: Vector Database has been successfully created and saved in the '{db_directory}' folder!")
    return vector_db

# --- Execution ---

if __name__ == "__main__":
    sample_file = "sample.docx"

    if os.path.exists(sample_file):
        vector_db = create_and_save_vector_db(sample_file)
    else:
        print(f"❌ File '{sample_file}' was not found!")