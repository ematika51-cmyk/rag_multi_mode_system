import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

folder_path = "./my_files"
documents = []

print("Files load ho rahi hain...")

# --- STEP 1: MULTI-FORMAT DATA INGESTION ---
if os.path.exists(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        
        # PDF files ke liye
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"Loaded PDF: {file}")
            
        # DOCX files ke liye
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
            print(f"Loaded DOCX: {file}")

print(f"\nTotal {len(documents)} document pages/files load ho gayi hain.")

if len(documents) > 0:
    # --- STEP 2: TEXT CHUNKING ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total {len(chunks)} chunks ban chuke hain.")

    # --- STEP 3: STYLOMETRY & STYLE GEOMETRY ---
    print("\nEmbedding model load ho raha hai...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def get_stylometric_features(text):
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        avg_sentence_len = len(words) / max(len(sentences), 1)
        unique_word_ratio = len(set(words)) / max(len(words), 1)
        
        return {
            "word_count": len(words),
            "avg_sentence_length": round(avg_sentence_len, 2),
            "vocabulary_richness": round(unique_word_ratio, 2)
        }

    def calculate_style_similarity(ref_text, gen_text):
        ref_vector = embedding_model.embed_query(ref_text)
        gen_vector = embedding_model.embed_query(gen_text)
        
        similarity_score = cosine_similarity([ref_vector], [gen_vector])[0][0]
        return round(float(similarity_score) * 100, 2)

    # Reference text from loaded files
    full_reference_text = " ".join([doc.page_content for doc in documents])

    # AI Sample Output for Comparison
    sample_generated_text = "This is a reference text sample to test stylometric similarity with uploaded files."

    # Analysis
    ref_stats = get_stylometric_features(full_reference_text)
    ai_stats = get_stylometric_features(sample_generated_text)
    style_score = calculate_style_similarity(full_reference_text, sample_generated_text)

    # Output Display
    print("\n================ RESULT ================")
    print(f"Reference File Stats: {ref_stats}")
    print(f"AI Output Stats: {ai_stats}")
    print(f"Style Match Score: {style_score}%")
    print("========================================")
else:
    print("Error: Files load nahi ho sakeen. Path check karein.")