import streamlit as st
import logging
import requests
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import numpy as np
import os
import chardet
from typing import List, Union

logging.basicConfig(level=logging.INFO)

# MongoDB setup
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["rag_db"]
collection = mongo_db["documents"]

class EmbeddingFunction:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, input_text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for input text."""
        try:
            if isinstance(input_text, str):
                input_text = [input_text]
            vectors = self.model.encode(input_text, convert_to_numpy=True)
            if len(vectors.shape) == 1:
                vectors = vectors.reshape(1, -1)
            return vectors
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise

# Initialize embedding model
embedding_model = EmbeddingFunction("paraphrase-multilingual-MiniLM-L12-v2")

def add_document_to_mongodb(documents: List[str], ids: List[str]) -> None:
    """Add documents with their embeddings to MongoDB."""
    try:
        for doc, doc_id in zip(documents, ids):
            if not doc.strip():
                raise ValueError("Cannot add an empty or whitespace-only document.")

            # Generate embedding and ensure it's a numpy array
            embedding_vector = embedding_model(doc)
            
            # Insert document with embedding
            collection.insert_one({
                "_id": doc_id,
                "document": doc,
                "embedding": embedding_vector[0].tolist()
            })
            logging.info(f"Successfully added document with ID: {doc_id}")
    except Exception as e:
        logging.error(f"Error adding document: {e}")
        raise

def query_documents_from_mongodb(query_text: str, n_results: int = 1) -> List[str]:
    """Query similar documents using cosine similarity."""
    try:
        query_embedding = embedding_model(query_text)
        docs = list(collection.find())
        
        if not docs:
            logging.warning("No documents found in the database")
            return []

        # Calculate similarities
        similarities = []
        for doc in docs:
            doc_embedding = np.array(doc["embedding"])
            similarity = np.dot(query_embedding[0], doc_embedding) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(doc_embedding)
            )
            similarities.append((similarity, doc))

        # Sort and return top results
        top_results = sorted(similarities, key=lambda x: x[0], reverse=True)[:n_results]
        return [doc["document"] for _, doc in top_results]
    except Exception as e:
        logging.error(f"Error querying documents: {e}")
        return []

def query_with_ollama(prompt: str, model_name: str) -> str:
    """Query Ollama model with error handling."""
    try:
        logging.info(f"Querying Ollama with model {model_name}")
        llm = OllamaLLM(model=model_name)
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        error_msg = f"Error with Ollama query: {str(e)}"
        logging.error(error_msg)
        return error_msg

def retrieve_and_answer(query_text: str, model_name: str) -> str:
    """Retrieve relevant documents and generate answer."""
    retrieved_docs = query_documents_from_mongodb(query_text)
    context = "\n\n".join(retrieved_docs) if retrieved_docs else "No relevant documents found."
    
    augmented_prompt = (
        f"Context: {context}\n\n"
        f"Question: {query_text}\n"
        f"Answer based on the context provided above:"
    )
    return query_with_ollama(augmented_prompt, model_name)

def fetch_text_from_url(url: str) -> str:
    """Fetch and extract text content from URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text(separator="\n")
        return "\n".join(line.strip() for line in text.split("\n") if line.strip())
    except Exception as e:
        logging.error(f"Error fetching text from URL: {e}")
        return ""

def main():
    st.set_page_config(page_title="AI Assistant for Constitution", layout="wide")
    st.title("AI Assistant for Constitution")

    model = "llama2:13b"  # Using a larger model for better responses
    
    menu = st.sidebar.selectbox(
        "Choose an action",
        ["Show Documents in MongoDB", "Add Document", "Upload File and Ask Question", 
         "Enter URL and Ask Question", "Ask a General Question"]
    )

    if menu == "Show Documents in MongoDB":
        st.subheader("Stored Documents in MongoDB")
        documents = list(collection.find())
        if documents:
            for i, doc in enumerate(documents, start=1):
                with st.expander(f"Document {i}"):
                    st.write(doc['document'])
        else:
            st.info("No documents available in the database.")

    elif menu == "Add Document":
        st.subheader("Add a New Document to MongoDB")
        
        col1, col2 = st.columns(2)
        with col1:
            new_doc = st.text_area("Enter the new document:", height=200)
        with col2:
            uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

        if st.button("Add Document", type="primary"):
            try:
                if uploaded_file:
                    file_bytes = uploaded_file.read()
                    detected_encoding = chardet.detect(file_bytes)['encoding']
                    file_content = file_bytes.decode(detected_encoding)
                    content = file_content
                elif new_doc.strip():
                    content = new_doc
                else:
                    st.warning("Please enter text or upload a file.")
                    return

                doc_id = f"doc_{collection.count_documents({}) + 1}"
                add_document_to_mongodb([content], [doc_id])
                st.success(f"Document successfully added with ID: {doc_id}")
            except Exception as e:
                st.error(f"Failed to add document: {str(e)}")

    elif menu == "Upload File and Ask Question":
        st.subheader("Upload a file and ask a question")
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

        if uploaded_file:
            try:
                file_bytes = uploaded_file.read()
                detected_encoding = chardet.detect(file_bytes)['encoding']
                file_content = file_bytes.decode(detected_encoding)

                with st.expander("View file content"):
                    st.text_area("Content", file_content, height=200)

                question = st.text_input("Ask a question about this content:")
                if question:
                    with st.spinner("Generating response..."):
                        response = query_with_ollama(
                            f"Context: {file_content}\n\nQuestion: {question}\nAnswer:",
                            model
                        )
                        st.write("Response:", response)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    elif menu == "Enter URL and Ask Question":
        st.subheader("Enter a URL and ask a question")
        url = st.text_input("Enter URL:")

        if url:
            with st.spinner("Fetching content..."):
                content = fetch_text_from_url(url)
                if content:
                    with st.expander("View extracted content"):
                        st.text_area("Content", content, height=200)

                    question = st.text_input("Ask a question about this content:")
                    if question:
                        with st.spinner("Generating response..."):
                            response = query_with_ollama(
                                f"Context: {content}\n\nQuestion: {question}\nAnswer:",
                                model
                            )
                            st.write("Response:", response)
                else:
                    st.error("Failed to fetch content from the URL.")

    elif menu == "Ask a General Question":
        st.subheader("Ask a General Question")
        query = st.text_input("Enter your question:")
        
        if query:
            with st.spinner("Searching and generating response..."):
                response = retrieve_and_answer(query, model)
                st.write("Response:", response)

if __name__ == "__main__":
    main()
