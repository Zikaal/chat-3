# AI Assistant for Constitution of Kazakhstan

This project is an AI-powered assistant designed to answer questions related to the **Constitution of the Republic of Kazakhstan**. The assistant utilizes advanced natural language processing techniques, MongoDB for data storage, and Streamlit for an interactive interface.

## Features

1. **Store Documents**
   - Add documents manually or upload `.txt` files.
   - Automatically generate vector embeddings for documents and store them in MongoDB.

2. **Search and Retrieve**
   - Query MongoDB to find the most relevant documents using vector similarity.

3. **Ask Questions**
   - Interact with the AI assistant to get answers based on:
     - Stored documents in MongoDB.
     - Uploaded files.
     - Content fetched from a provided URL.

4. **Fetch Content from URLs**
   - Provide a URL (e.g., to the Constitution) to extract and analyze its content.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start MongoDB server:
   - Ensure MongoDB is running locally on `mongodb://localhost:27017/`.

4. Run the application:
   ```bash
   streamlit run <script_name>.py
   ```

## Usage

### Menu Options

- **Show Documents in MongoDB**: View stored documents and their content.
- **Add Document**: Add new documents manually or upload `.txt` files.
- **Upload File and Ask Question**: Upload a file, view its content, and ask questions about it.
- **Enter URL and Ask Question**: Provide a URL, extract its content, and ask questions about it.
- **Ask a General Question**: Ask questions based on all stored documents in MongoDB.

### Supported File Formats
- `.txt` (for document upload).

## Example Workflow

1. Start the application with `streamlit run`.
2. Navigate to **Enter URL and Ask Question**.
3. Input the URL of the Constitution.
4. Extract and view the content.
5. Ask a question, such as: *"What are the guaranteed rights?"*.
6. View the AI-generated response.

## Requirements

- Python 3.8+
- MongoDB running locally
- Dependencies listed in `requirements.txt`

## Project Structure

```
.
├── src/
│   ├── main.py          # Main application script
├── test/                # Tests for functionality
|   ├── main.py          
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── LICENSE              # License information
```

## References

1. [Constitution of the Republic of Kazakhstan](https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912)
2. [LangChain Documentation](https://python.langchain.com)
3. [Streamlit Documentation](https://docs.streamlit.io)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
