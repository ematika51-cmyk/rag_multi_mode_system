import os
import fitz 
import markdownify
import streamlit as st

from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Page Configuration
st.set_page_config(page_title="RAG Multi-Mode System", layout="wide")

st.title("RAG System (Directory Ingestion & Dual-Mode)")

# Embeddings & Chroma Path Setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
PERSIST_DIR = "./chroma_db"
FOLDER_PATH = "./my_files"

# Sidebar Setup
st.sidebar.header("Configuration")
user_groq_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your gsk_... key")

st.sidebar.markdown("---")
st.sidebar.header("Mode Selection")
enable_style_mimic = st.sidebar.toggle("Enable Style Mimicking Mode", value=False)

if enable_style_mimic:
    st.sidebar.info("Mode: Style Mimicking\nGenerates response matching the document's tone and style.")
else:
    st.sidebar.info("Mode: Standard QA\nProvides direct and precise factual answers.")


def load_pdfs_as_markdown(folder_path):
    documents = []
    if not os.path.exists(folder_path):
        return documents

    # Cover page indicators filter out karne ke liye words
    ignore_keywords = ["SUBMITTED TO", "SUBMITTED BY", "REG NO", "COMSATS UNIVERSITY", "ATTOCK CAMPUS"]

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            doc = fitz.open(pdf_path)
            full_markdown = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                html_content = page.get_text("html")
                md_text = markdownify.markdownify(html_content, heading_style="ATX")
                
                # Check agar pehla page cover/title page hai to usay ignore karein
                upper_md = md_text.upper()
                is_cover_page = sum(1 for kw in ignore_keywords if kw in upper_md) >= 2
                
                if not is_cover_page:
                    full_markdown += f"\n\n<!-- Page {page_num + 1} -->\n\n" + md_text

            if full_markdown.strip():
                documents.append(
                    Document(page_content=full_markdown, metadata={"source": file_name, "format": "markdown"})
                )
    return documents


@st.cache_resource(show_spinner=False)
def setup_vector_database(folder_path):
    if not os.path.exists(folder_path) or len(os.listdir(folder_path)) == 0:
        return None, 0, f"Error: '{folder_path}' folder is empty or not found."

    docs = []
    
    # 1. Load PDFs (Filtered)
    pdf_docs = load_pdfs_as_markdown(folder_path)
    docs.extend(pdf_docs)

    # 2. Load DOCX
    try:
        docx_loader = DirectoryLoader(folder_path, glob="**/*.docx", loader_cls=Docx2txtLoader)
        docx_docs = docx_loader.load()
        docs.extend(docx_docs)
    except Exception:
        pass

    if not docs:
        return None, 0, f"Error: No valid PDF or DOCX files found in '{folder_path}'."

    # Optimal Chunking Strategy for Academic Notes
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # Clean any accidental remaining metadata chunks
    cleaned_chunks = []
    for chunk in chunks:
        content_upper = chunk.page_content.upper()
        if not ("SUBMITTED TO" in content_upper and "REG NO" in content_upper):
            cleaned_chunks.append(chunk)

    # Vector Store Storage
    vector_db = Chroma.from_documents(cleaned_chunks, embeddings, persist_directory=PERSIST_DIR)
    return vector_db, len(cleaned_chunks), None


# Initialize Database
with st.spinner("Processing documents from 'my_files'..."):
    vector_db, num_chunks, err = setup_vector_database(FOLDER_PATH)

if err:
    st.sidebar.error(err)
else:
    st.sidebar.success(f"Status: Ready | Total Chunks: {num_chunks}")


# Main Query Interface
input_label = "Enter Topic for Paragraph Generation:" if enable_style_mimic else "Enter Question for Standard RAG QA:"
user_input = st.text_input(input_label, placeholder="Type here...")

if st.button("Submit", type="primary"):
    if not user_groq_key.strip():
        st.error("Please enter a valid Groq API Key in the sidebar.")
    elif not user_input.strip():
        st.warning("Please enter a query or topic.")
    elif vector_db is None:
        st.error("Vector database is not initialized.")
    else:
        with st.spinner("Retrieving context and generating response..."):
            try:
                # Top 5 most relevant chunks (Fixed from k=100)
                retriever = vector_db.as_retriever(search_kwargs={"k": 8})
                retrieved_docs = retriever.invoke(user_input)
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

                with st.expander("View Retrieved Context Chunks"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        source_name = doc.metadata.get("source", "Unknown Source")
                        st.markdown(f"**Chunk {i} (Source: {source_name}):**")
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
                        Match the exact writing style, tone, and language mix of the context below.
                        
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
                        You are a precise AI assistant. Answer the question accurately based strictly on the provided context.
                        If the context does not contain enough information, state that clearly.
                        
                        QUESTION: {topic}
                        
                        CONTEXT:
                        ---
                        {context}
                        ---
                        
                        ANSWER:
                        """
                    )

                chain = prompt | llm
                response = chain.invoke({"topic": user_input, "context": context_text})

                output_title = "Style-Matched Output:" if enable_style_mimic else "QA Output:"
                st.subheader(output_title)
                st.write(response.content)

            except Exception as e:
                st.error(f"Error: {e}")