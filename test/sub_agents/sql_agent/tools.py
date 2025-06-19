import os
import json
import sqlite3
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Database setup
DB_PATH = "./data/db/medical.db"  # Update this path as needed

def _referenced_tables(sql: str) -> List[str]:
    """Extract table names appearing after FROM or JOIN."""
    pattern = r"\bfrom\s+([\w\"\.]+)|\bjoin\s+([\w\"\.]+)"
    refs = [grp for m in re.finditer(pattern, sql, flags=re.I) for grp in m.groups() if grp]
    return [r.strip('"').split('.')[-1] for r in refs]

def _is_sql_valid(sql: str, allowed: List[str]) -> bool:
    """Return True iff every referenced table is in allowed list."""
    return all(t in allowed for t in _referenced_tables(sql))

def _get_schema_info() -> str:
    """Return schema (table, columns, sample categorical values)."""
    try:
        info_lines: List[str] = []
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # List user-defined tables
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = [row["name"] for row in cursor.fetchall()]

            for table in tables:
                info_lines.append(f"\nTable: {table}")
                cursor.execute(f"PRAGMA table_info('{table}')")
                columns = cursor.fetchall()

                for col in columns:
                    col_name = col["name"]
                    col_type = col["type"] or "UNKNOWN"
                    samples = ""

                    # For text-like columns, fetch up to 10 distinct sample values
                    if col_type.upper() in ("TEXT", ""):
                        try:
                            cursor.execute(
                                f"SELECT DISTINCT {col_name} FROM {table} WHERE {col_name} IS NOT NULL LIMIT 10"
                            )
                            vals = [str(r[0]) for r in cursor.fetchall()]
                            if vals:
                                samples = f" (sample values: {', '.join(vals)})"
                        except sqlite3.OperationalError:
                            samples = " (error fetching sample values)"

                    info_lines.append(f"  - {col_name} ({col_type}){samples}")

        return "\n".join(info_lines)
    except Exception as e:
        return """
Table: med
  - name (TEXT) (sample values: Bobby Jackson, Leslie Terry, Danny Smith)
  - age (INTEGER)
  - gender (TEXT) (sample values: Male, Female)
  - blood_type (TEXT) (sample values: B-, A+, A-, O+, AB+)
  - medical_condition (TEXT) (sample values: Cancer, Obesity, Diabetes)
  - date_of_admission (TIMESTAMP)
  - doctor (TEXT) (sample values: Matthew Smith, Samantha Davies)
  - hospital (TEXT) (sample values: Sons and Miller, Kim Inc)
  - insurance_provider (TEXT) (sample values: Blue Cross, Medicare, Aetna)
  - billing_amount (REAL)
  - room_number (INTEGER)
  - admission_type (TEXT) (sample values: Urgent, Emergency, Elective)
  - discharge_date (TIMESTAMP)
  - medication (TEXT) (sample values: Paracetamol, Ibuprofen, Aspirin)
  - test_results (TEXT) (sample values: Normal, Inconclusive, Abnormal)
"""

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
    """Executes the given SQL query and returns the result as a list of dicts."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        data = [dict(r) for r in rows]
        return data
    except Exception as e:
        raise Exception(f"SQL execution error: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

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
    Execute a SQL query and return formatted results.
    
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
    """Get the database schema information."""
    print("Retrieving schema information...", '*'*50)
    return _get_schema_info() 
