import os
import fitz  # PyMuPDF
import markdownify
import streamlit as st

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Page Setup
st.set_page_config(page_title="RAG Multi-Mode System", page_icon="⚡", layout="wide")

st.title("⚡ RAG System (PDF-to-Markdown Ingestion & Dual-Mode)")

# Embeddings & Chroma Path
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
PERSIST_DIR = "./chroma_db"

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
user_groq_key = st.sidebar.text_input("Groq API Key", type="password", help="Paste your gsk_... key here")

# --- TOGGLE SWITCH ---
st.sidebar.markdown("---")
st.sidebar.header("🔀 Mode Selection")
enable_style_mimic = st.sidebar.toggle("Enable Style Mimicking Mode", value=False)

if enable_style_mimic:
    st.sidebar.info("🎭 Mode Active: Style Mimicking\nGenerates content matching the tone & style of the document.")
else:
    st.sidebar.info("🎯 Mode Active: Standard QA\nDirect RAG retrieval & precise factual answers.")


# --- FUNCTION: CONVERT PDF TO MARKDOWN ---
def convert_pdf_to_markdown(pdf_path):
    doc = fitz.open(pdf_path)
    full_markdown = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        html_content = page.get_text("html")
        md_text = markdownify.markdownify(html_content, heading_style="ATX")
        full_markdown += f"\n\n<!-- Page {page_num + 1} -->\n\n" + md_text
    return full_markdown


# --- DOCUMENT INGESTION FUNCTION ---
def setup_vector_database():
    docs = []
    file_name = ""
    file_type = ""

    if os.path.exists("sample.pdf"):
        file_name = "sample.pdf"
        file_type = "PDF (Converted to Markdown)"
        # PDF ko Markdown text mein badlein
        md_text = convert_pdf_to_markdown(file_name)
        docs = [Document(page_content=md_text, metadata={"source": file_name, "format": "markdown"})]
    elif os.path.exists("sample.docx"):
        file_name = "sample.docx"
        file_type = "DOCX Ingestion"
        loader = Docx2txtLoader(file_name)
        docs = loader.load()
    else:
        return None, 0, "Error: 'sample.pdf' ya 'sample.docx' file nahi mili!", "", ""

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Vector DB Storage
    vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    return vector_db, len(chunks), None, file_name, file_type


# Process Document
with st.spinner("Processing document into Markdown & setting up Vector DB..."):
    vector_db, num_chunks, err, active_file, active_type = setup_vector_database()

if err:
    st.sidebar.error(err)
else:
    st.sidebar.success(f"✅ Loaded: `{active_file}`")
    st.sidebar.caption(f"Method: {active_type} | Total Chunks: {num_chunks}")


# --- MAIN UI & QUERY ---
input_label = "Enter Topic for Paragraph Generation:" if enable_style_mimic else "Enter Question for Standard RAG QA:"
user_input = st.text_input(input_label, placeholder="e.g., What is Retrieval Augmented Generation?")

if st.button("🚀 Process Request", type="primary"):
    if not user_groq_key.strip():
        st.error("⚠️ Please paste a valid Groq API Key (starts with gsk_) in the sidebar!")
    elif not user_input.strip():
        st.warning("Please enter a query or topic first.")
    elif vector_db is None:
        st.error("Vector database is not initialized.")
    else:
        with st.spinner("Retrieving context and generating output..."):
            try:
                retriever = vector_db.as_retriever(search_kwargs={"k": 2})
                retrieved_docs = retriever.invoke(user_input)
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

                with st.expander(f"🔍 View Retrieved Reference Chunks from {active_file}"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc.page_content)
                        st.divider()

                llm = ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    groq_api_key=user_groq_key.strip(),
                    temperature=0.7 if enable_style_mimic else 0.2
                )

                if enable_style_mimic:
                    prompt = PromptTemplate.from_template(
                        """
                        You are an expert content writer and style mimicker.
                        
                        TASK:
                        Write a paragraph on the topic: "{topic}"
                        
                        INSTRUCTION:
                        Match the EXACT writing style, tone, and language mix (e.g. Roman Urdu / English) of the context below.
                        
                        CONTEXT:
                        ---
                        {context}
                        ---
                        
                        GENERATED PARAGRAPH:
                        """
                    )
                else:
                    prompt = PromptTemplate.from_template(
                        """
                        You are a precise AI assistant. Answer the user question accurately based ONLY on the provided context.
                        If the context does not contain enough information, state that clearly.
                        
                        QUESTION: {topic}
                        
                        CONTEXT FROM DOCUMENT:
                        ---
                        {context}
                        ---
                        
                        ANSWER:
                        """
                    )

                chain = prompt | llm
                response = chain.invoke({"topic": user_input, "context": context_text})

                output_title = "✍️ Style-Matched Output:" if enable_style_mimic else "📝 QA Output:"
                st.subheader(output_title)
                st.write(response.content)

            except Exception as e:
                st.error(f"❌ Error: {e}")