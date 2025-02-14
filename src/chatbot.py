import streamlit as st
import logging
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import numpy as np
import chardet
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

# Подключение к базе данных MongoDB
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["rag_db"]
collection = mongo_db["documents"]

class EmbeddingFunction:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        vectors = self.model.encode(input)
        if len(vectors) == 0:
            raise ValueError("Empty embedding generated.")
        return vectors

embedding = EmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")

def add_document_to_mongodb(documents, ids):
    try:
        for doc, doc_id in zip(documents, ids):
            if not doc.strip():
                raise ValueError("Cannot add an empty or whitespace-only document.")

            embedding_vector = embedding(doc)
            logging.info(f"Generated embedding for document '{doc}': {embedding_vector}")

            collection.insert_one({
                "_id": doc_id,
                "document": doc,
                "embedding": embedding_vector[0].tolist()
            })
    except Exception as e:
        logging.error(f"Error adding document: {e}")
        raise
def query_documents_from_mongodb(query_text, n_results=1):
    try:
        query_embedding = embedding(query_text)[0]
        docs = collection.find()

        similarities = []
        for doc in docs:
            doc_embedding = np.array(doc["embedding"])
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((similarity, doc))

        top_results = sorted(similarities, key=lambda x: x[0], reverse=True)[:n_results]
        return [doc["document"] for _, doc in top_results]
    except Exception as e:
        logging.error(f"Error querying documents: {e}")
        return []

def query_with_ollama(prompt, model_name):
    try:
        logging.info(f"Sending prompt to Ollama with model {model_name}: {prompt}")
        llm = OllamaLLM(model=model_name)
        response = llm.invoke(prompt)
        logging.info(f"Ollama response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error with Ollama query: {e}")
        return f"Error with Ollama API: {e}"
def retrieve_and_answer(query_text, model_name):
    retrieved_docs = query_documents_from_mongodb(query_text)
    context = " ".join(retrieved_docs) if retrieved_docs else "No relevant documents found."

    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    return query_with_ollama(augmented_prompt, model_name)


# Функция для извлечения текста с сайта Конституции Республики Казахстан
def get_constitution_text():
    url = "https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        constitution_text = ""
        for paragraph in soup.find_all("p"):
            constitution_text += paragraph.get_text() + "\n"
        logging.info(f"Extracted Constitution text: {constitution_text[:500]}...")  # Покажем первые 500 символов
        return constitution_text
       
    else:
        logging.error("Error fetching the Constitution text from the website.")
        return "Error fetching the Constitution text from the website."

st.title("Chat with Ollama")

model = "llama3.2:1b"
menu = st.sidebar.selectbox("Choose an action", ["Show Documents in MongoDB", "Add New Document to MongoDB as Vector", "Upload File and Ask Question", "Ask Ollama a Question", "Ask Question About Constitution"])

if menu == "Show Documents in MongoDB":
    st.subheader("Stored Documents in MongoDB")
    documents = collection.find()
    if documents:
        for i, doc in enumerate(documents, start=1):
            st.write(f"{i}. {doc['document']}")
    else:
        st.write("No data available!")

elif menu == "Add New Document to MongoDB as Vector":
    st.subheader("Add a New Document to MongoDB")
    new_doc = st.text_area("Enter the new document:")
    uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

    if st.button("Add Document"):
        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()
                detected_encoding = chardet.detect(file_bytes)['encoding']
                if not detected_encoding:
                    raise ValueError("Failed to detect file encoding.")
                file_content = file_bytes.decode(detected_encoding)

                doc_id = f"doc{collection.count_documents({}) + 1}"
                st.write(f"Adding document from file: {uploaded_file.name}")
                add_document_to_mongodb([file_content], [doc_id])
                st.success(f"Document added successfully with ID {doc_id}")
            except Exception as e:
                st.error(f"Failed to add document: {e}")
        elif new_doc.strip(): 
            try:
                doc_id = f"doc{collection.count_documents({}) + 1}"
                st.write(f"Adding document: {new_doc}")
                add_document_to_mongodb([new_doc], [doc_id])
                st.success(f"Document added successfully with ID {doc_id}")
            except Exception as e:
                st.error(f"Failed to add document: {e}")
        else:
            st.warning("Please enter a non-empty document or upload a file before adding.")

elif menu == "Upload File and Ask Question":
    st.subheader("Upload a file and ask a question about its content")
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            detected_encoding = chardet.detect(file_bytes)['encoding']
            if not detected_encoding:
                raise ValueError("Failed to detect file encoding.")
            file_content = file_bytes.decode(detected_encoding)

            st.write("File content successfully loaded:")
            st.text_area("File Content", file_content, height=200)

            question = st.text_input("Ask a question about this file's content:")
            if question:
                response = query_with_ollama(f"Context: {file_content}\n\nQuestion: {question}\nAnswer:", model)
                st.write("Response:", response)

        except Exception as e:
            st.error(f"Failed to process the file: {e}")

elif menu == "Ask Ollama a Question":
    query = st.text_input("Ask a question")
    if query:
        response = retrieve_and_answer(query, model)
        st.write("Response:", response)



if menu == "Ask Question About Constitution":
    question = st.text_input("Ask a question about the Constitution of Kazakhstan")
    if question:
        try:
            # Извлекаем текст с сайта Конституции
            constitution_text = get_constitution_text()
            if constitution_text:
                # Ограничиваем объем текста для передачи в модель (например, первые 2000 символов)
                context = constitution_text[:2000]  # Ограничение на первые 2000 символов
                logging.info(f"Constitution text: {context[:500]}...")  # Отладочный вывод

                # Формируем запрос к Ollama
                augmented_prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
                response = query_with_ollama(augmented_prompt, model)

                # Выводим ответ от Ollama и краткий текст Конституции
                st.write("Constitution Text (Extract):")
                st.text_area("Constitution Text", context, height=200)
                st.write("Response from Ollama:", response)

                # Генерация краткого текста
                summary_prompt = f"Summarize the following content: {context}"
                summary = query_with_ollama(summary_prompt, model)
                st.write("Summary of the Constitution Text:", summary)
            else:
                st.write("Failed to fetch Constitution text.")
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            st.write("An error occurred while processing the request.")
