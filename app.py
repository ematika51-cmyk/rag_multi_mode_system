import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

st.set_page_config(page_title="RAG Stylometry & Generation Pipeline", layout="wide")
st.title("📄 RAG Semantic Retrieval, Stylometry & Free LLM Generation")

folder_path = "./my_files"
chroma_dir = "./chroma_db"
documents = []

st.sidebar.header("Pipeline Status")

# ==========================================
# 1. DOCUMENT INGESTION (Skipping Temp Files)
# ==========================================
if os.path.exists(folder_path):
    for file in os.listdir(folder_path):
        if file.startswith("~$"):
            continue

        file_path = os.path.join(folder_path, file)
        
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            st.sidebar.warning(f"Error loading {file}: {e}")

if len(documents) > 0:
    st.sidebar.success(f"Loaded {len(documents)} pages/documents.")

    # ==========================================
    # 2. TEXT CHUNKING
    # ==========================================
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    st.sidebar.info(f"Generated {len(chunks)} text chunks.")

    # ==========================================
    # 3. EMBEDDINGS & CHROMADB VECTOR STORE SETUP (MiniLM - FREE)
    # ==========================================
    @st.cache_resource
    def setup_vector_store(_chunks):
        # 100% Free Local MiniLM Embeddings Model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(
            documents=_chunks, 
            embedding=embeddings,
            persist_directory=chroma_dir
        )
        return embeddings, vectorstore

    embedding_model, vectorstore = setup_vector_store(chunks)

    # ==========================================
    # 4. FREE LOCAL TEXT GENERATOR SETUP (NO API KEY)
    # ==========================================
    @st.cache_resource
    def load_local_generator():
        # Open-source free model (No OpenAI/Groq keys required)
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=250,
            temperature=0.1
        )
        return HuggingFacePipeline(pipeline=pipe)

    llm = load_local_generator()

    # Stylometry helper function
    def get_stylometric_features(text):
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        avg_sentence_len = len(words) / max(len(sentences), 1)
        unique_word_ratio = len(set(words)) / max(len(words), 1)
        return {
            "Word Count": len(words),
            "Avg Sentence Length": round(avg_sentence_len, 2),
            "Vocabulary Richness": round(unique_word_ratio, 2)
        }

    full_reference_text = " ".join([doc.page_content for doc in documents])
    ref_stats = get_stylometric_features(full_reference_text)

    # UI Metrics
    st.subheader("📊 Document Stylometry Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Word Count", ref_stats["Word Count"])
    col2.metric("Avg Sentence Length", ref_stats["Avg Sentence Length"])
    col3.metric("Vocabulary Richness", ref_stats["Vocabulary Richness"])

    st.markdown("---")

    # ==========================================
    # 5. RETRIEVAL & GENERATION UI
    # ==========================================
    st.subheader("🔍 Search, Retrieval & Free Generation")
    user_query = st.text_input("Enter your question/topic:", value="Explain the concept of memory management")

    if st.button("Search & Generate Response"):
        with st.spinner("Retrieving from ChromaDB & Generating Answer..."):
            # ChromaDB Vector Search (MiniLM Embeddings)
            retrieved_docs = vectorstore.similarity_search(user_query, k=3)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Local Free Generator Prompt
            prompt = f"Answer the following question based ONLY on the provided context.\n\nContext:\n{context_text}\n\nQuestion: {user_query}\nAnswer:"
            generated_answer = llm.invoke(prompt)

            # Style Similarity Match
            ref_vector = embedding_model.embed_query(full_reference_text)
            query_vector = embedding_model.embed_query(user_query)
            similarity_score = cosine_similarity([ref_vector], [query_vector])[0][0]
            style_score = round(float(similarity_score) * 100, 2)

        st.subheader("🤖 Generated Response (Local Open-Source Model)")
        st.success(generated_answer)

        st.write(f"**Query Style Similarity Match:** `{style_score}%`")

        st.subheader("📚 Top Retrieved Relevant Context Chunks (from ChromaDB)")
        for i, doc in enumerate(retrieved_docs):
            with st.expander(f"Relevant Chunk {i+1} (Source: {os.path.basename(doc.metadata.get('source', 'N/A'))})"):
                st.write(doc.page_content)

else:
    st.error("No valid PDF or DOCX files found in './my_files' directory!")