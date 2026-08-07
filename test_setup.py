from langchain_community.embeddings import HuggingFaceEmbeddings

print("Loading Embedding Model...")
# MiniLM model initialization
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Test text embedding
sample_text = "RAG system with LangChain and ChromaDB"
query_result = embeddings.embed_query(sample_text)

print("Setup Successful! Vector dimension:", len(query_result))