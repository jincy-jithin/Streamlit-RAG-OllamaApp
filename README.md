# Local RAG Chatbot with LangChain, Ollama and Streamlit

This project demonstrates a Retrieval-Augmented Generation (RAG) chatbot that answers questions about President Biden's 2023 State of the Union speech.

The application runs entirely locally using Ollama and uses a vector database to retrieve relevant context before generating responses.

## Features

- Local LLM using Ollama
- Vector search using Chroma
- Embeddings from HuggingFace
- Streamlit web interface
- Modern LangChain LCEL pipeline
- Maximal Marginal Relevance (MMR) retrieval

## Requirements

Install Ollama:

https://ollama.com

Pull the model:

ollama pull phi3

Install dependencies using uv:

uv sync

## Run the Application

Start the Streamlit app:

streamlit run app.py

The app will be available at:

http://localhost:8501

## Example Questions

- What were the main topics in the State of the Union speech?
- What did the president say about the economy?
- What policies were proposed?

## Tech Stack

- LangChain
- Ollama
- Chroma
- HuggingFace Embeddings
- Streamlit