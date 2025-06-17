# sql_agent.py
"""
SQL Agent with real database logic using Gemini for NL to SQL conversion
Uses chat protocols for communication between agents
"""

import glob
import os
import json
import sqlite3
import logging
import re
from datetime import datetime, timezone

from uuid import uuid4
from typing import Any, Dict, List

import google.generativeai as genai
from uagents import Agent, Protocol, Context
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)
from dotenv import load_dotenv
import requests  # Added for ASI:ONE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GENAI_API_KEY:
    logger.critical("GOOGLE_API_KEY environment variable not set.")
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# ASI:ONE API setup
ASI_ONE_API_KEY = os.getenv('ASI_ONE_API_KEY')
if not ASI_ONE_API_KEY:
    logger.critical("ASI_ONE_API_KEY environment variable not set.")
    raise ValueError("ASI_ONE_API_KEY environment variable not set.")
ASI_ONE_URL = "https://api.asi1.ai/v1/chat/completions"
ASI_ONE_MODEL = "asi1-mini"  # or another model if desired
ASI_ONE_TEMPERATURE = 0.7  # Set your desired temperature here
ASI_ONE_MAX_TOKENS = 300  # Adjust as needed

# Database setup
db_files = glob.glob("./data/db/*.db")
if not db_files:
    logger.critical("No .db files found in ./data/db")
    raise FileNotFoundError("No .db files found in ./data/db")
DB_PATH = db_files[0]
logger.info(f"Using SQLite database at: {DB_PATH}")

# Configure Gemini
genai.configure(api_key=GENAI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# ---------- AGENT SETUP ----------
sql_agent = Agent(
    name="sqlAgent", 
    mailbox=True,
    port=9001,
    endpoint=["http://localhost:9001/submit"]
)

chat_proto = Protocol(spec=chat_protocol_spec)

# ---------- HELPER FUNCTIONS ----------
def _referenced_tables(sql: str) -> List[str]:
    """Extract table names appearing after FROM or JOIN (very naive)."""
    pattern = r"\\bfrom\\s+([\\w\\\"\\.]+)|\\bjoin\\s+([\\w\\\"\\.]+)"
    refs = [grp for m in re.finditer(pattern, sql, flags=re.I) for grp in m.groups() if grp]
    return [r.strip('\"').split('.')[-1] for r in refs]

def _is_sql_valid(sql: str, allowed: List[str]) -> bool:
    """Return True iff every referenced table is in allowed list."""
    return all(t in allowed for t in _referenced_tables(sql))

def _get_schema_info() -> str:
    """Return schema (table, columns, sample categorical values)."""
    try:
        info_lines: List[str] = []
        logger.debug("Connecting to SQLite DB to extract schema info.")
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # List user-defined tables
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = [row["name"] for row in cursor.fetchall()]
            logger.debug(f"Found tables: {tables}")

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
                        except sqlite3.OperationalError as e:
                            samples = " (error fetching sample values)"
                            logger.warning(f"Error fetching sample values for {table}.{col_name}: {e}")

                    info_lines.append(f"  - {col_name} ({col_type}){samples}")

        logger.info("Schema info extraction successful.")
        return "\n".join(info_lines)
    except Exception as e:
        logger.error(f"Failed to extract schema info: {e}")
        return """
Table: med
  - name (TEXT) (sample values: Bobby JacksOn, LesLie TErRy, DaNnY sMitH, andrEw waTtS, adrIENNE bEll)
  - age (INTEGER)
  - gender (TEXT) (sample values: Male, Female)
  - blood_type (TEXT) (sample values: B-, A+, A-, O+, AB+)
  - medical_condition (TEXT) (sample values: Cancer, Obesity, Diabetes)
  - date_of_admission (TIMESTAMP)
  - doctor (TEXT) (sample values: Matthew Smith, Samantha Davies, Tiffany Mitchell, Kevin Wells, Kathleen Hanna)
  - hospital (TEXT) (sample values: Sons and Miller, Kim Inc, Cook PLC, Hernandez Rogers and Vang,, White-White)
  - insurance_provider (TEXT) (sample values: Blue Cross, Medicare, Aetna)
  - billing_amount (REAL)
  - room_number (INTEGER)
  - admission_type (TEXT) (sample values: Urgent, Emergency, Elective)
  - discharge_date (TIMESTAMP)
  - medication (TEXT) (sample values: Paracetamol, Ibuprofen, Aspirin, Penicillin)
  - test_results (TEXT) (sample values: Normal, Inconclusive, Abnormal)
"""

def _clean_sql_response(sql_query: str) -> str:
    """Cleans Gemini's SQL output and ensures safety."""
    logger.debug(f"Cleaning SQL response: {sql_query!r}")
    if sql_query.startswith("```sql"):
        sql_query = sql_query[6:]
    if sql_query.startswith("```"):
        sql_query = sql_query[3:]
    if sql_query.endswith("```"):
        sql_query = sql_query[:-3]
    sql_query = sql_query.strip()

    sql_lower = sql_query.lower()
    if not sql_lower.startswith("select"):
        logger.error("Generated query does not start with SELECT.")
        raise Exception("Generated query must be a SELECT statement")

    dangerous = ["drop", "delete", "insert", "update", "alter", "create"]
    for kw in dangerous:
        if kw in sql_lower:
            logger.error(f"Dangerous keyword detected in query: {kw}")
            raise Exception(f"Dangerous keyword in query: {kw}")
    
    logger.debug(f"Cleaned SQL query: {sql_query}")
    return sql_query

def _generate_fallback_sql(nl_query: str) -> str:
    """Very loose heuristics when Gemini is silent or errors out."""
    q = nl_query.lower()
    logger.warning(f"Using fallback SQL for query: {nl_query!r}")

    if any(kw in q for kw in ("count", "how many", "total")):
        return "SELECT COUNT(*) AS total_records FROM med;"
    
    if any(kw in q for kw in ("billing", "amount", "cost", "expense")):
        return "SELECT Name, Billing_Amount FROM med ORDER BY Billing_Amount DESC LIMIT 25;"
    
    if "medication" in q or "drug" in q:
        return "SELECT Name, Medication, Medical_Condition FROM med WHERE Medication IS NOT NULL LIMIT 50;"
    
    if "age" in q or "oldest" in q or "youngest" in q:
        return "SELECT Name, Age, Gender, Medical_Condition FROM med ORDER BY Age DESC LIMIT 50;"
    
    if "admission" in q or "admitted" in q:
        return "SELECT Name, Date_of_Admission AS Admission_Date, Admission_Type FROM med ORDER BY Date_of_Admission DESC LIMIT 25;"
    
    return "SELECT * FROM med LIMIT 10;"

def asi_one_generate(prompt: str) -> str:
    payload = json.dumps({
        "model": ASI_ONE_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": ASI_ONE_TEMPERATURE,
        "stream": False,
        "max_tokens": ASI_ONE_MAX_TOKENS
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {ASI_ONE_API_KEY}'
    }
    try:
        response = requests.post(ASI_ONE_URL, headers=headers, data=payload)
        data = response.json()
        if 'choices' in data and len(data['choices']) > 0:
            message = data['choices'][0].get('message', {})
            content = message.get('content', '')
            return content.strip()
        else:
            return "No choices found in ASI:ONE response."
    except Exception as e:
        logger.error(f"ASI:ONE API error: {e}")
        return "An error occurred while generating the answer with ASI:ONE."

def convert_to_sql(natural_language_query: str) -> str:
    """Convert a natural-language question into a valid SQLite SELECT statement."""
    logger.info(f"Converting NL query to SQL: {natural_language_query!r}")

    schema_info = _get_schema_info()
    base_prompt = f"""You are an expert SQL generator for hospital patient-record analytics. Convert the user's question into a VALID SQLite SELECT query.

SCHEMA:
{schema_info}

QUESTION: \"{natural_language_query}\"

RULES:
1. Output ONLY the SQL (no markdown or comments).
2. Use the exact column names shown.
3. The ONLY table is \"med\". No joins to other tables.
4. The query MUST start with SELECT and never modify data.

SQL query:"""

    allowed_tables = ["med"]

    try:
        # First attempt with ASI:ONE
        logger.debug("ASI:ONE prompt (attempt 1)")
        sql_query = asi_one_generate(base_prompt)
        sql_query = _clean_sql_response(sql_query)

        # Validate: check referenced tables
        if not _is_sql_valid(sql_query, allowed_tables):
            bad_tables = _referenced_tables(sql_query)
            logger.warning(f"ASI:ONE used invalid tables {bad_tables}; retrying.")

            # Second attempt (hard reminder)
            retry_prompt = (
                base_prompt
                + "\n\nIMPORTANT: You MUST use only the table \"med\". "
                  "Do NOT invent or join other tables. Regenerate now."
            )
            sql_query = asi_one_generate(retry_prompt)
            sql_query = _clean_sql_response(sql_query)

        # Final validation
        if not _is_sql_valid(sql_query, allowed_tables):
            raise ValueError("ASI:ONE still produced invalid tables after retry.")

        logger.info(f"Generated SQL: {sql_query}")
        return sql_query

    except Exception as e:
        logger.error(f"SQL generation failed or invalid: {e}. Falling back.")
        return _generate_fallback_sql(natural_language_query)

def _execute_sql(sql_query: str) -> List[Dict[str, Any]]:
    """Executes the given SQL query and returns the result as a list of dicts."""
    logger.info(f"Executing SQL query: {sql_query}")
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        data = [dict(r) for r in rows]
        logger.info(f"SQL execution returned {len(data)} rows.")
        return data
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

def _prepare_data_summary(data: List[Dict[str, Any]]) -> str:
    """Prepares a summary of the data for answer generation."""
    logger.debug(f"Preparing data summary for {len(data)} records.")
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

def generate_answer(original_query: str, sql_query: str, data: List[Dict[str, Any]]) -> str:
    """Generates a natural language answer using ASI:ONE."""
    data_summary = _prepare_data_summary(data)
    prompt = f"""Based on the following data, generate a helpful and natural language answer to the original question.

Original Question: \"{original_query}\"
SQL Query Executed: \"{sql_query}\"
Data Retrieved: {data_summary}

ANSWER REQUIREMENTS:
1. Provide a concise, direct answer.
2. Integrate values from 'Data Retrieved'.
3. If one value only, embed it in a sentence.
4. If list/table, summarise clearly.
5. Do NOT include the SQL query.
6. If 'No data found', say exactly that.

Answer:"""
    try:
        logger.debug("Sending answer generation prompt to ASI:ONE")
        answer = asi_one_generate(prompt)
        logger.info("ASI:ONE answer generation successful.")
        return answer
    except Exception as e:
        logger.error(f"ASI:ONE answer generation failed: {e}")
        return "An error occurred while generating the answer with ASI:ONE."

# ---------- STARTUP HANDLER ----------
@sql_agent.on_event("startup")
async def startup_handler(ctx: Context):
    ctx.logger.info(f"SQL Agent started - Name: {ctx.agent.name}, Address: {ctx.agent.address}")

# ---------- MESSAGE HANDLER ----------
@chat_proto.on_message(ChatMessage)
async def handle_sql_query(ctx: Context, sender: str, msg: ChatMessage):
    for item in msg.content:
        if isinstance(item, TextContent):
            try:
                # Parse request_id and query from message
                request_id, query_text = item.text.split(":::", 1)
                ctx.logger.info(f"SQLAgent: Received query: {query_text} (Request ID: {request_id})")

                # 1. Convert NL to SQL
                sql_query = convert_to_sql(query_text)
                ctx.logger.info(f"SQLAgent: Generated SQL query: {sql_query}")

                # 2. Execute SQL
                try:
                    data = _execute_sql(sql_query)
                except Exception as e:
                    ctx.logger.exception("SQL execution failed.")
                    error_resp = f"SQLAgent error: {e}"
                    
                    # Send error response back
                    error_msg = ChatMessage(
                        timestamp=datetime.now(timezone.utc),
                        msg_id=uuid4(),
                        content=[TextContent(type="text", text=f"{request_id}:::{error_resp}")],
                    )
                    await ctx.send(sender, error_msg)
                    return

                # 3. Generate natural language answer
                answer_text = generate_answer(query_text, sql_query, data)
                ctx.logger.info(f"SQLAgent: Generated answer for request {request_id}")

                # 4. Send response back
                response_msg = ChatMessage(
                    timestamp=datetime.now(timezone.utc),
                    msg_id=uuid4(),
                    content=[TextContent(type="text", text=f"{request_id}:::{answer_text}")],
                )
                await ctx.send(sender, response_msg)
                ctx.logger.info(f"SQLAgent: Sent response for {request_id} back to sender")

            except ValueError:
                ctx.logger.error(f"Malformed message format: {item.text}")
            except Exception as e:
                ctx.logger.exception(f"Error processing message: {e}")

            # Send acknowledgment
            ack = ChatAcknowledgement(
                timestamp=datetime.now(timezone.utc),
                acknowledged_msg_id=msg.msg_id
            )
            await ctx.send(sender, ack)

# ---------- ACK HANDLER ----------
@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")

sql_agent.include(chat_proto, publish_manifest=True)

if __name__ == '__main__':
    sql_agent.run()