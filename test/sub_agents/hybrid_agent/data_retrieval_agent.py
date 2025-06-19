from google.adk.agents import Agent
from typing import Dict, Any
import sqlite3
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from pathlib import Path
import google.generativeai as genai

# Ensure required directories exist
DB_DIR = Path("./data/db")
VECTORSTORE_DIR = Path("./vectorstores")
DB_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

def execute_sql_query(query: str) -> Dict[str, Any]:
    """Execute SQL query and return results."""
    db_path = DB_DIR / "medical.db"
    if not db_path.exists():
        return {
            "status": "error",
            "message": "Database file not found. Please ensure the database exists at ./data/db/medical.db"
        }
    
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        return {
            "status": "success",
            "data": [dict(r) for r in rows],
            "message": f"Successfully retrieved {len(rows)} records"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"SQL execution error: {str(e)}"
        }
    finally:
        try:
            conn.close()
        except:
            pass

def similarity_search(query: str) -> Dict[str, Any]:
    """Perform similarity search on FAISS database, loading and merging all stores like document_agent."""
    print(f"Performing similarity search for query: {query}")
    try:
        DB_FOLDER = str(VECTORSTORE_DIR)
        os.makedirs(DB_FOLDER, exist_ok=True)
        stores = []
        for dir_name in os.listdir(DB_FOLDER):
            path = os.path.join(DB_FOLDER, dir_name)
            if os.path.isdir(path):
                try:
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                        return {
                            "status": "error",
                            "message": "GOOGLE_API_KEY environment variable not set",
                            "data": []
                        }
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=api_key
                    )
                    store = FAISS.load_local(
                        path, embeddings, allow_dangerous_deserialization=True
                    )
                    stores.append(store)
                except Exception as e:
                    print(f"Error loading store from {path}: {e}")
        if not stores:
            return {
                "status": "warning",
                "message": "No documents available in the vector store.",
                "data": []
            }
        base = stores[0]
        for other in stores[1:]:
            base.merge_from(other)
        vectorstore = base
        docs = vectorstore.similarity_search(query, k=5)
        print(f"Found {len(docs)} documents matching the query.")
        return {
            "status": "success",
            "data": [doc.page_content for doc in docs],
            "message": f"Found {len(docs)} relevant documents"
        }
    except Exception as e:
        print(f"Similarity search error: {str(e)}")
        return {
            "status": "error",
            "message": f"Similarity search error: {str(e)}",
            "data": []
        }

def synthesize_answer(user_query: str, sql_data: str, doc_data: str) -> str:
    prompt = f"""
You are a medical assistant. Use BOTH the database results and the document excerpts to answer the user's question.

User Question: "{user_query}"

Database Results:
{sql_data or 'No relevant data found in the database.'}

Document Excerpts:
{doc_data or 'No relevant documents found.'}

Instructions:
- Use both sources if possible.
- If one source is empty, answer from the other and note the gap.
- Provide a clear, helpful, and concise answer.
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# Helper for SQL summary (reuse or adapt from your _prepare_data_summary logic)
def _prepare_data_summary(data):
    if not data:
        return "No data found"
    if isinstance(data, list):
        if len(data) == 1 and len(data[0]) == 1:
            key = list(data[0].keys())[0]
            return f"The direct answer value is: {data[0][key]}"
        summary_parts = [f"Total Records: {len(data)}"]
        if len(data) <= 5:
            summary_parts.append(f"All Data:\n{data}")
        else:
            summary_parts.append(f"Sample Data (first 5 rows):\n{data[:5]}")
            summary_parts.append(f"[... {len(data) - 5} more rows]")
        cols = ", ".join(data[0].keys())
        summary_parts.append(f"Columns: {cols}")
        return "\n".join(summary_parts)
    return str(data)

def retrieve_data_and_similarity(sql_query: str, user_query: str) -> str:
    sql_result = execute_sql_query(sql_query)
    sim_result = similarity_search(user_query)

    # Prepare user-friendly summaries
    sql_summary = _prepare_data_summary(sql_result["data"]) if sql_result["status"] == "success" else sql_result["message"]
    doc_summary = "\n---\n".join(sim_result["data"]) if sim_result["status"] == "success" and sim_result["data"] else sim_result["message"]

    # Synthesize final answer
    answer = synthesize_answer(user_query, sql_summary, doc_summary)
    return answer

data_retrieval_agent = Agent(
    name="data_retrieval_agent",
    model="gemini-2.0-flash",
    description="Agent for retrieving data from both SQL and vector databases",
    instruction="""
    You are a data retrieval agent that handles both SQL queries and similarity searches.
    Your role is to execute queries and return formatted results.

    When you receive a request:
    1. Use the retrieve_data_and_similarity tool, which takes both a SQL query and the original user query.
    2. It will execute the SQL query and perform a similarity search, then return a combined, user-friendly answer.
    3. If the database or document store is missing or empty, provide a clear, helpful message and answer from whichever source is available.

    Focus on:
    - Executing queries safely
    - Handling both SQL and vector search results
    - Formatting results clearly
    - Providing detailed error messages when things go wrong
    - Checking for required resources (database, vector store) before operations

    Always maintain data privacy and format sensitive information appropriately.
    If a resource is missing (database or vector store), provide a clear message about what's missing.
    """,
    sub_agents=[],
    tools=[retrieve_data_and_similarity],
) 