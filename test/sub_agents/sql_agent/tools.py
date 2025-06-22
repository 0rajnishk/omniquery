import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

# BigQuery setup
BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", "your-project-id")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "your-dataset-id")
bq_client = bigquery.Client(project=BQ_PROJECT_ID)

def _referenced_tables(sql: str) -> List[str]:
    """Extract table names appearing after FROM or JOIN."""
    pattern = r"\bfrom\s+([\w\"\.]+)|\bjoin\s+([\w\"\.]+)"
    refs = [grp for m in re.finditer(pattern, sql, flags=re.I) for grp in m.groups() if grp]
    return [r.strip('"').split('.')[-1] for r in refs]

def _is_sql_valid(sql: str, allowed: List[str]) -> bool:
    """Return True iff every referenced table is in allowed list."""
    return all(t in allowed for t in _referenced_tables(sql))

def _get_schema_info() -> str:
    """Return schema (table, columns, sample categorical values) from BigQuery."""
    try:
        info_lines: List[str] = []
        # List tables in the dataset
        tables = list(bq_client.list_tables(BQ_DATASET_ID))
        for table in tables:
            table_ref = table.table_id
            info_lines.append(f"\nTable: {table_ref}")
            # Get table schema
            table_obj = bq_client.get_table(f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_ref}")
            for field in table_obj.schema:
                col_name = field.name
                col_type = field.field_type
                samples = ""
                # Try to get up to 10 distinct sample values for STRING fields
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

def _clean_sql_response(sql_query: str) -> str:
    """Cleans SQL output and ensures safety."""
    if sql_query.startswith("```sql"):
        sql_query = sql_query[6:]
    if sql_query.startswith("```"):
        sql_query = sql_query[3:]
    if sql_query.endswith("```"):
        sql_query = sql_query[:-3]
    sql_query = sql_query.strip()

    sql_lower = sql_query.lower()
    if not sql_lower.startswith("select"):
        raise Exception("Generated query must be a SELECT statement")

    dangerous = ["drop", "delete", "insert", "update", "alter", "create"]
    for kw in dangerous:
        if kw in sql_lower:
            raise Exception(f"Dangerous keyword in query: {kw}")
    
    return sql_query

def _execute_sql(sql_query: str) -> List[Dict[str, Any]]:
    """Executes the given SQL query on BigQuery and returns the result as a list of dicts."""
    try:
        query_job = bq_client.query(sql_query)
        rows = list(query_job)
        data = [dict(row.items()) for row in rows]
        return data
    except Exception as e:
        raise Exception(f"BigQuery execution error: {e}")

def _prepare_data_summary(data: List[Dict[str, Any]]) -> str:
    """Prepares a summary of the data for answer generation."""
    if not data:
        return "No data found"

    if len(data) == 1 and len(data[0]) == 1:
        key = list(data[0].keys())[0]
        return f"The direct answer value is: {data[0][key]}"

    summary_parts = [f"Total Records: {len(data)}"]
    if len(data) <= 5:
        summary_parts.append(f"All Data:\n{json.dumps(data, indent=2)}")
    else:
        summary_parts.append(f"Sample Data (first 5 rows):\n{json.dumps(data[:5], indent=2)}")
        summary_parts.append(f"[... {len(data) - 5} more rows]")
    cols = ", ".join(data[0].keys())
    summary_parts.append(f"Columns: {cols}")
    return "\n".join(summary_parts)

def execute_sql_query(query: str) -> str:
    """
    Execute a SQL query and return formatted results from BigQuery.
    
    Args:
        query: The SQL query to execute
        
    Returns:
        Formatted results as a string
    """
    print(f"Executing SQL query: {query}", '*'*50)
    try:
        # Clean and validate SQL
        sql_query = _clean_sql_response(query)
        
        # Execute query
        data = _execute_sql(sql_query)
        print(f"Query executed successfully, retrieved {len(data)} rows.", '*'*50)
        # Prepare summary
        return _prepare_data_summary(data)
    except Exception as e:
        return f"Error executing query: {str(e)}"

def get_schema() -> str:
    """Get the database schema information from BigQuery."""
    print("Retrieving schema information from BigQuery...", '*'*50)
    return _get_schema_info() 
