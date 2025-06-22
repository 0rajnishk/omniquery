# Hybrid Agent

The Hybrid Agent combines SQL data retrieval from BigQuery with document similarity search using FAISS and Google Gemini embeddings. It synthesizes answers using both structured and unstructured data.

## Components
- `agent.py`: Main agent definition and workflow.
- `tools.py`: Functions for schema retrieval (using BigQuery) and SQL query generation.
- `data_retrieval_agent.py`: Executes SQL queries on BigQuery and performs similarity search.

## Environment Variables
- `GOOGLE_API_KEY`: Google Gemini API key
- `BQ_PROJECT_ID`: Your Google Cloud project ID
- `BQ_DATASET_ID`: Your BigQuery dataset ID

## Usage
- Answers questions using both database and document sources
- Retrieves schema and data from BigQuery
- Performs document similarity search
- Synthesizes a combined answer 