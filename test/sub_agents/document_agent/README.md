# Document Agent

The Document Agent specializes in processing and analyzing medical documents. It uses FAISS for similarity search and Google Gemini embeddings to find relevant information in uploaded PDFs.

## Components
- `agent.py`: Main agent definition and workflow.
- `tools.py`: Functions for document similarity search using FAISS and Gemini embeddings.

## Environment Variables
- `GOOGLE_API_KEY`: Google Gemini API key

## Usage
- Finds and summarizes relevant information from medical documents
- Answers questions based on document content 