import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def run_rag_pipeline(user_topic):

    print("Step 1: Loading Vector Database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )

    print(f"\nStep 2: Retrieving context for topic '{user_topic}'...")
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = retriever.invoke(user_topic)

    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print(f"✅ Found {len(retrieved_docs)} relevant context chunks!")

    groq_api_key = ""
    
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key= "",
        temperature=0.7
    )

    style_prompt = PromptTemplate.from_template(
        """
        You are an expert content writer and style mimicker.
        
        TASK:
        Write a paragraph on the following topic: "{topic}"
        
        STRICT WRITING STYLE INSTRUCTIONS:
        1. Analyze the writing style, tone, vocabulary, and structure of the reference context below.
        2. Write the new paragraph about the topic using the EXACT SAME writing style, language mix (e.g., Roman Urdu/English), and tone as the reference context.
        
        REFERENCE CONTEXT FROM DOCUMENTS:
        ---
        {context}
        ---
        
        GENERATED PARAGRAPH:
        """
    )

    print("\nStep 3: Groq LLM is generating text with matching writing style...\n")
    chain = style_prompt | llm
    response = chain.invoke({"topic": user_topic, "context": context_text})

    return response.content

if __name__ == "__main__":
    user_topic = "what is vector database in RAG??"
    
    try:
        generated_output = run_rag_pipeline(user_topic)
        print("================ GENERATED OUTPUT ================")
        print(generated_output)
        print("==================================================")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")