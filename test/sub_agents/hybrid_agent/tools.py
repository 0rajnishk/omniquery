from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

# BigQuery setup
BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", "your-project-id")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "your-dataset-id")
bq_client = bigquery.Client(project=BQ_PROJECT_ID)

def get_schema_info() -> str:
    """Return schema information for SQL query generation from BigQuery."""
    try:
        info_lines: List[str] = []
        tables = list(bq_client.list_tables(BQ_DATASET_ID))
        for table in tables:
            table_ref = table.table_id
            info_lines.append(f"\nTable: {table_ref}")
            table_obj = bq_client.get_table(f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_ref}")
            for field in table_obj.schema:
                col_name = field.name
                col_type = field.field_type
                samples = ""
                if col_type.upper() == "STRING":
                    try:
                        query = f"SELECT DISTINCT `{col_name}` FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_ref}` WHERE `{col_name}` IS NOT NULL LIMIT 10"
                        query_job = bq_client.query(query)
                        vals = [str(row[col_name]) for row in query_job]
                        if vals:
                            samples = f" (sample values: {', '.join(vals)})"
                    except Exception:
                        samples = " (error fetching sample values)"
                info_lines.append(f"  - {col_name} ({col_type}){samples}")
        return "\n".join(info_lines)
    except Exception as e:
        return f"Error retrieving schema from BigQuery: {e}"

def generate_sql_query(natural_language_query: str) -> str:
    """
    Generate SQL query based on natural language and schema.
    This is a simplified version - in practice, you'd use a more sophisticated NL to SQL conversion.
    """
    schema = get_schema_info()
    # The actual SQL generation would be handled by the agent's model
    return f"Generated SQL query for: {natural_language_query}" 