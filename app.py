
from pathlib import Path
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = Path(__file__).parent
document_path = BASE_DIR/"president-bidens-state-of-the-union-2023.txt"
chroma_db_path = BASE_DIR / "chroma_db"

# -------------------------
# Load or Create Vector DB or create if not available 
# -------------------------
@st.cache_resource
def create_or_load_vectordb():

    # Ensure the source document exists
    if not document_path.exists():
        st.error(f"Document not found: {document_path}")
        st.stop()

     # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------------------------
    # If DB already exists → load it
    # -------------------------
    if chroma_db_path.exists():
        vector_db = Chroma(
            persist_directory = chroma_db_path,
            embedding_function = embeddings
        )
        return vector_db 
    
    # -------------------------
    # Otherwise create DB
    # -------------------------
    else:
        loader = TextLoader(document_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 100
            )
        docs = splitter.split_documents(documents)
        vector_db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory = chroma_db_path)
        return vector_db

# -------------------------
# Build RAG Chain
# -------------------------
@st.cache_resource
def build_rag_chain():

    # Initialize the language model with the specified model
    llm = ChatOllama(
        model="phi3",
        temperature=0
    )
    vector_db = create_or_load_vectordb()
    retriever = vector_db.as_retriever(
        search_type = "mmr",
        search_kwargs = {
            "k":5,
            "fetch_k": 20
        }

    )
    # Define a template for generating answers using provided context
    prompt_template = PromptTemplate.from_template (
     """You are a helpful assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context:
{context}

Question:
{question}

Provide a concise answer under 200 words.
"""
    )
   #takes retrieved documents,extracts their text,joins them into one string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    
    rag_chain = (
        {
            "context": retriever | format_docs, #sends to the LLM as context
            "question": RunnablePassthrough()
        } # Pass the context and question
        | prompt_template # Format the prompt using the custom RAG prompt template
        | llm # Use the language model to generate a response
        | StrOutputParser() # Parse the output to a string
  
    )
    return rag_chain

# -------------------------
# Streamlit UI
# -------------------------
st.title("Bidens State of the Union 2023 — RAG Chatbot")
st.write("Ask questions about President Biden's 2023 State of the Union speech.")

query = st.text_input("Ask a question")

if query:
    rag_chain = build_rag_chain()

    with st.spinner("Thinking..."):
        response = rag_chain.invoke(query)

    st.write(response)


