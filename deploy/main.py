import asyncio
import os
import uuid
import logging
import shutil
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, APIRouter, Form, Request, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import json
from datetime import datetime, timezone
import re
import secrets
import sys
import argparse
import sqlite3
import pandas as pd

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

import google.generativeai as genai
from utils import add_user_query_to_history, call_agent_async
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from test.agent import test_agent

load_dotenv()

import os
print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))
# ─── Logging Setup ─────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── FastAPI Setup ─────────────────────────────────────────────────
app = FastAPI(title="Agent Flow Query API", version="1.0.0")
router = APIRouter(prefix="/document", tags=["documents"])


# Mount static files (if you have JS, CSS, etc. later)
app.mount("/static", StaticFiles(directory="static"), name="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Data Models ───────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

class RemoteDbDetails(BaseModel):
    type: str
    host: str
    port: int
    database: str
    username: str
    password: str

# ─── File Paths ────────────────────────────────────────────────────
# Use /tmp as the base directory for all data and vectorstores
TMP_BASE = "/tmp"
QUERY_FILE_PATH = os.path.join(TMP_BASE, "query_to_agents.txt")
RESPONSE_FILE_PATH = os.path.join(TMP_BASE, "response_from_agents.txt")
DB_FOLDER = os.path.join(TMP_BASE, "vectorstores")
DATA_FOLDER = os.path.join(TMP_BASE, "data")
DB_DATA_FOLDER = os.path.join(DATA_FOLDER, "db")
os.makedirs(DB_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(DB_DATA_FOLDER, exist_ok=True)

# ─── Google Gemini Embeddings ──────────────────────────────────────

GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your Google API key.")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GENAI_API_KEY
)

# ─── PDF Processor ─────────────────────────────────────────────────
def process_pdf(pdf_path: str, pdf_id: str):
    logger.info(f"Starting PDF processing for: {pdf_path} with id: {pdf_id}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents from PDF.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(os.path.join(DB_FOLDER, pdf_id))
    logger.info(f"Vectorstore saved locally with id: {pdf_id}")
    return {"success": True, "message": "PDF processed successfully"}

# ─── In-Memory Document Manager ────────────────────────────────────
class DocManager:
    def __init__(self):
        self.docs = {}  # doc_id: {filename, path}
        self.load_from_vectorstores()

    def load_from_vectorstores(self):
        # Scan the vectorstores folder for directories/files named as 'originalname---id'
        if not os.path.exists(DB_FOLDER):
            return
        for entry in os.listdir(DB_FOLDER):
            # Only consider directories (vectorstore format)
            entry_path = os.path.join(DB_FOLDER, entry)
            if os.path.isdir(entry_path):
                # Match pattern: originalname---id
                match = re.match(r"(.+)---([a-fA-F0-9\-]+)$", entry)
                if match:
                    original_name, doc_id = match.groups()
                    # Try to find the original file in data/db or data
                    possible_file = os.path.join(DATA_FOLDER, f"{doc_id}_{original_name}")
                    if not os.path.exists(possible_file):
                        # fallback: just use the name
                        possible_file = original_name
                    self.docs[doc_id] = {"filename": original_name, "path": possible_file}

    def upload_document(self, files: List[UploadFile]):
        for f in files:
            doc_id = str(uuid.uuid4())
            # Save with id in filename for traceability
            save_path = os.path.join(DB_FOLDER, f"{f.filename}---{doc_id}")
            # Save the uploaded file in data folder for reference
            data_file_path = os.path.join(DATA_FOLDER, f"{doc_id}_{f.filename}")
            with open(data_file_path, "wb") as dest:
                shutil.copyfileobj(f.file, dest)
            # Process and save vectorstore with the new naming
            process_pdf(data_file_path, f"{f.filename}---{doc_id}")
            self.docs[doc_id] = {"filename": f.filename, "path": data_file_path}

    def get_document(self, doc_id: str):
        # Try in memory first
        if doc_id in self.docs:
            return self.docs[doc_id]
        # Try to find in vectorstores
        for entry in os.listdir(DB_FOLDER):
            match = re.match(r"(.+)---([a-fA-F0-9\-]+)$", entry)
            if match:
                original_name, found_id = match.groups()
                if found_id == doc_id:
                    possible_file = os.path.join(DATA_FOLDER, f"{doc_id}_{original_name}")
                    if not os.path.exists(possible_file):
                        possible_file = original_name
                    self.docs[doc_id] = {"filename": original_name, "path": possible_file}
                    return self.docs[doc_id]
        return None

    def get_all_documents(self):
        # Always refresh from vectorstores
        self.load_from_vectorstores()
        return self.docs

    def delete_document(self, doc_id: str):
        meta = self.docs.pop(doc_id, None)
        if meta:
            # Remove vectorstore directory
            for entry in os.listdir(DB_FOLDER):
                match = re.match(r"(.+)---([a-fA-F0-9\-]+)$", entry)
                if match and match.group(2) == doc_id:
                    entry_path = os.path.join(DB_FOLDER, entry)
                    shutil.rmtree(entry_path, ignore_errors=True)
            # Remove data file
            data_file = os.path.join(DATA_FOLDER, f"{doc_id}_{meta['filename']}")
            if os.path.exists(data_file):
                try:
                    os.remove(data_file)
                except Exception:
                    pass
        return meta is not None

doc = DocManager()



# ─── Index Route ───────────────────────────────────────────────────
@app.get("/")
def get_index():
    return FileResponse("static/index.html")
# ─── Index Route ───────────────────────────────────────────────────
@app.get("/info")
def get_index():
    return FileResponse("static/info.html")

# # ─── Admin Route ───────────────────────────────────────────────────
# @app.get("/admin")
# def get_admin():
#     return FileResponse("static/admin.html")


# ─── Simple Admin Auth ─────────────────────────────────────────────
ADMIN_ID = "admin"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Super_Uagents")  # Change this in production!
current_token = None  # Stores the current valid session token

def get_token_from_request(request: Request):
    token = request.query_params.get("token")
    return token

def require_admin_token(request: Request):
    global current_token
    token = get_token_from_request(request)
    if not token or token != current_token:
        raise HTTPException(status_code=401, detail="Invalid or missing token. Please login again.")

@app.post("/login")
def login(admin_id: str = Form(...), password: str = Form(...)):
    global current_token
    if admin_id == ADMIN_ID and password == ADMIN_PASSWORD:
        current_token = secrets.token_urlsafe(32)
        return {"token": current_token}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials.")


# ─── Document Endpoints ────────────────────────────────────────────
@router.post("/")
async def upload_document(request: Request, files: List[UploadFile] = File(...), _: None = Depends(require_admin_token)):
    logger.info(f"Uploading {len(files)} file(s): {[file.filename for file in files]}")
    doc.upload_document(files)
    return {"message": "Upload document", "files": [file.filename for file in files]}

@router.get("/{document_id}")
async def get_document(document_id: str):
    logger.info(f"Retrieving document with ID: {document_id}")
    document = doc.get_document(document_id)
    if document is None:
        logger.warning(f"Document with ID '{document_id}' not found.")
        return {"message": f"Document {document_id} not found"}
    logger.info(f"Document with ID '{document_id}' retrieved successfully.")
    return document

@router.get("/")
async def get_all_documents():
    logger.info("Retrieving all documents.")
    documents = doc.get_all_documents()
    if not documents:
        logger.warning("No documents found.")
        return {"message": "No documents found"}
    logger.info(f"Retrieved {len(documents)} documents.")
    return documents

@router.delete("/{document_id}")
async def delete_document(request: Request, document_id: str, _: None = Depends(require_admin_token)):
    logger.info(f"Deleting document with ID: {document_id}")
    result = doc.delete_document(document_id)
    if result:
        logger.info(f"Document with ID '{document_id}' deleted successfully.")
        return {"message": f"Document {document_id} deleted"}
    else:
        logger.warning(f"Document with ID '{document_id}' not found.")
        return {"message": f"Document {document_id} not found"}


# ===== Session and Runner Setup (Singleton) =====
session_service = None
runner = None
SESSION_ID = None
USER_ID = "test_user"
APP_NAME = "Test Agent System"
initial_state = {
    "user_name": "Test User",
    "interaction_history": [],
}

async def ensure_session_and_runner():
    global session_service, runner, SESSION_ID
    if session_service is None:
        session_service = InMemorySessionService()
    if SESSION_ID is None:
        new_session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            state=initial_state,
        )
        SESSION_ID = new_session.id
    if runner is None:
        runner = Runner(
            agent=test_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    await ensure_session_and_runner()
    user_input = request.query
    # Update interaction history
    await add_user_query_to_history(
        session_service, APP_NAME, USER_ID, SESSION_ID, user_input
    )
    # Call the agent and get the response
    response = await call_agent_async(runner, USER_ID, SESSION_ID, user_input)
    if not response:
        raise HTTPException(status_code=500, detail="No response from agent.")
    return QueryResponse(response=response)

@app.post("/query/stream")
async def process_query_stream(request: QueryRequest):
    await ensure_session_and_runner()
    user_input = request.query
    
    # Update interaction history
    await add_user_query_to_history(
        session_service, APP_NAME, USER_ID, SESSION_ID, user_input
    )
    
    async def generate_stream():
        try:
            # Call the agent and get the streaming response for all queries
            async for chunk in call_agent_stream_async(runner, USER_ID, SESSION_ID, user_input):
                if chunk:
                    # Send the chunk as a Server-Sent Event
                    chunk_data = {'chunk': chunk, 'type': 'chunk'}
                    yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Send end signal
            end_data = {'type': 'end'}
            yield f"data: {json.dumps(end_data)}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            error_data = {'error': str(e), 'type': 'error'}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

# Helper function for streaming agent responses
async def call_agent_stream_async(runner, user_id: str, session_id: str, user_input: str):
    """Stream the agent response asynchronously"""
    try:
        # This is a placeholder - you'll need to implement the actual streaming
        # based on your agent's capabilities
        response = await call_agent_async(runner, user_id, session_id, user_input)
        
        if response:
            # Stream the response in larger chunks to preserve formatting
            lines = response.split('\n')
            for line in lines:
                if line.strip():  # Only send non-empty lines
                    yield line + '\n'
                    # Add a small delay to simulate real streaming
                    await asyncio.sleep(0.1)
        else:
            yield "No response available.\n"
            
    except Exception as e:
        logger.error(f"Error in agent streaming: {e}")
        yield f"Error: {str(e)}\n"

# ─── New Endpoints ─────────────────────────────────────────────────
def sanitize(name: str) -> str:
    """
    Make a string safe for use as a SQL table name:
      • trim spaces
      • lower-case
      • spaces → underscores
      • drop characters that aren't alphanumeric or "_"
    """
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    return re.sub(r"[^\w]", "", name)

def sanitize_column(col: str) -> str:
    """
    Sanitize a column name:
      - Replace spaces with underscores
      - If there is a capital letter followed by a small letter, insert underscore before capital
      - Convert to lower-case
      - Remove non-alphanumeric/underscore
    """
    # Replace spaces with underscores
    col = re.sub(r"\s+", "_", col)
    # Insert underscore before capital letters that are followed by lowercase (for CamelCase)
    col = re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', col)
    # Convert to lower-case
    col = col.lower()
    # Remove non-alphanumeric/underscore
    col = re.sub(r"[^\w]", "", col)
    return col

def excel_to_sqlite(file_path: str, db_path: str) -> str:
    """Read Excel/CSV file and write an SQLite database.  
    Returns the path to the database created."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Create db directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Determine file type and process accordingly
    file_ext = os.path.splitext(file_path)[1].lower()
    
    with sqlite3.connect(db_path) as conn:
        if file_ext in ['.xlsx', '.xls']:
            # Handle Excel files
            excel = pd.ExcelFile(file_path)
            for sheet in excel.sheet_names:
                df = excel.parse(sheet)
                table_name = sanitize(sheet)

                # Sanitize column names
                df.columns = [sanitize_column(col) for col in df.columns]

                # Write DataFrame → SQL table (replace if it already exists)
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                logger.info(f"  → Sheet '{sheet}' saved as table '{table_name}'")
        
        elif file_ext == '.csv':
            # Handle CSV files
            df = pd.read_csv(file_path)
            table_name = sanitize(os.path.splitext(os.path.basename(file_path))[0])
            
            # Sanitize column names
            df.columns = [sanitize_column(col) for col in df.columns]
            
            # Write DataFrame → SQL table (replace if it already exists)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            logger.info(f"  → CSV saved as table '{table_name}'")
        
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    logger.info(f"SQLite database created at: {os.path.abspath(db_path)}")
    return db_path

@app.post("/upload-csv")
async def upload_csv(request: Request, file: UploadFile = File(...), _: None = Depends(require_admin_token)):
    # Save the uploaded Excel/CSV file temporarily
    csv_folder = os.path.join(DB_DATA_FOLDER, "csv")
    os.makedirs(csv_folder, exist_ok=True)
    csv_name = file.filename
    temp_file_path = os.path.join(csv_folder, csv_name)
    
    # Save the uploaded file temporarily
    with open(temp_file_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)
    logger.info(f"Uploaded file saved temporarily as {temp_file_path}")

    try:
        # Convert to SQLite database
        db_folder = DB_DATA_FOLDER
        db_path = os.path.join(db_folder, "medical.db")
        
        # Convert Excel/CSV to SQLite
        excel_to_sqlite(temp_file_path, db_path)
        
        logger.info(f"Excel/CSV file converted to SQLite database: {db_path}")
        
        return {"message": "Excel/CSV file converted to SQLite database successfully", "filename": csv_name, "db_path": db_path}
        
    except Exception as e:
        logger.error(f"Error converting file to SQLite: {e}")
        raise HTTPException(status_code=500, detail=f"Error converting file to SQLite: {str(e)}")
    
    finally:
        # Clean up temporary file
        try:
            os.remove(temp_file_path)
            logger.info(f"Temporary file removed: {temp_file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {temp_file_path}: {e}")

@app.post("/update-remote-db")
async def update_remote_db(request: Request, details: RemoteDbDetails, _: None = Depends(require_admin_token)):
    # Save the remote DB details as a JSON file in /tmp/data/
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    unique_name = f"remote_db_{timestamp}_{uuid.uuid4().hex}.json"
    save_path = os.path.join(DATA_FOLDER, unique_name)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(details.model_dump(), f, indent=2)
    logger.info(f"Remote DB details saved as {save_path}")
    return {"message": "Remote DB details saved successfully", "filename": unique_name}


# ─── Mount Document Router ─────────────────────────────────────────
app.include_router(router)

# ─── Run Server ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    if os.path.exists(QUERY_FILE_PATH):
        os.remove(QUERY_FILE_PATH)
    if os.path.exists(RESPONSE_FILE_PATH):
        os.remove(RESPONSE_FILE_PATH)

    logging.info("Starting FastAPI application (separate process)...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
