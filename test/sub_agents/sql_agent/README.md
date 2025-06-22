# SQL Agent

The SQL Agent is responsible for generating and executing SQL queries based on natural language questions. It uses Google BigQuery to retrieve schema information and execute queries on your dataset.

## Components
- `agent.py`: Main agent definition and workflow.
- `tools.py`: Functions for schema retrieval and SQL execution (using BigQuery).
- `query_executor_agent.py`: Sub-agent for running SQL queries and formatting results.
- `schema_agent.py`: Sub-agent for providing database schema information.

## Environment Variables
- `BQ_PROJECT_ID`: Your Google Cloud project ID
- `BQ_DATASET_ID`: Your BigQuery dataset ID

## Usage
- Converts user questions to SQL queries
- Retrieves schema and data from BigQuery
- Returns clear, formatted answers 