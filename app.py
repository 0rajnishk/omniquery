import os
import re
import shutil
import secrets
import sys
import argparse
import sqlite3
import pandas as pd
import asyncio
import logging
from datetime import datetime, timezone
import uuid
from typing import List

from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Request, StreamingResponse, APIRouter
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from uagents import Agent, Context, Protocol
from uagents.setup import fund_agent_if_low
from uagents.storage import InMemorySessionService
from uagents.runner import Runner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DB_FOLDER = "vectorstores"
DATA_FOLDER = "data"
QUERY_FILE_PATH = "query.txt"
RESPONSE_FILE_PATH = "response.txt"

# Create necessary directories
os.makedirs(DB_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="OmniQuery API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize router
router = APIRouter(prefix="/document", tags=["documents"])

# Pydantic models
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

# PDF Processing function
def process_pdf(pdf_path: str, pdf_id: str):
    """Process PDF and create vector store"""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        splits = text_splitter.split_documents(pages)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Save vectorstore
        store_path = os.path.join(DB_FOLDER, f"{os.path.splitext(os.path.basename(pdf_path))[0]}---{pdf_id}")
        vectorstore.save_local(store_path)
        
        logger.info(f"Vector store saved to {store_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        return False

# Document Manager
class DocManager:
    def __init__(self):
        self.docs = {}
        self.load_from_vectorstores()

    def load_from_vectorstores(self):
        # Scan the vectorstores folder for directories/files named as 'originalname---id'
        for entry in os.listdir(DB_FOLDER):
            match = re.match(r"(.+)---([a-fA-F0-9\-]+)$", entry)
            if match:
                original_name, doc_id = match.groups()
                self.docs[doc_id] = {"filename": original_name + ".pdf"}

    def upload_document(self, files: List[UploadFile]):
        for file in files:
            if file.filename.endswith('.pdf'):
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                # Save file
                file_path = os.path.join(DATA_FOLDER, f"{doc_id}_{file.filename}")
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                # Process PDF
                if process_pdf(file_path, doc_id):
                    self.docs[doc_id] = {"filename": file.filename}
                    logger.info(f"Document uploaded successfully: {file.filename}")

    def get_document(self, doc_id: str):
        # Try in memory first
        if doc_id in self.docs:
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
            data_file = os.path.join(os.path.dirname(__file__), "data", f"{doc_id}_{meta['filename']}")
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

@app.get("/info")
def get_index():
    return FileResponse("static/info.html")

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

# Placeholder functions for agent interaction
async def add_user_query_to_history(session_service, app_name, user_id, session_id, user_input):
    # Placeholder implementation
    pass

async def call_agent_async(runner, user_id, session_id, user_input):
    # Placeholder implementation - replace with actual agent call
    return f"This is a placeholder response to: {user_input}"

async def call_agent_stream_async(runner, user_id, session_id, user_input):
    """Stream the agent response asynchronously"""
    try:
        # This is a placeholder - you'll need to implement the actual streaming
        # based on your agent's capabilities
        response = await call_agent_async(runner, user_id, session_id, user_input)
        
        if response:
            # Simulate streaming by sending the response in chunks
            words = response.split()
            for i, word in enumerate(words):
                yield word + " "
                # Add a small delay to simulate real streaming
                await asyncio.sleep(0.05)
        else:
            yield "No response available."
            
    except Exception as e:
        logger.error(f"Error in agent streaming: {e}")
        yield f"Error: {str(e)}"

# Placeholder agent
test_agent = Agent(name="test_agent")

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
            # Call the agent and get the streaming response
            async for chunk in call_agent_stream_async(runner, USER_ID, SESSION_ID, user_input):
                if chunk:
                    # Send the chunk as a Server-Sent Event
                    yield f"data: {json.dumps({'chunk': chunk, 'type': 'chunk'})}\n\n"
            
            # Send end signal
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"data: {json.dumps({'error': str(e), 'type': 'error'})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

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
    csv_folder = os.path.join(os.path.dirname(__file__), "data", "csv")
    os.makedirs(csv_folder, exist_ok=True)
    csv_name = file.filename
    temp_file_path = os.path.join(csv_folder, csv_name)
    
    # Save the uploaded file temporarily
    with open(temp_file_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)
    logger.info(f"Uploaded file saved temporarily as {temp_file_path}")

    try:
        # Convert to SQLite database
        db_folder = os.path.join(os.path.dirname(__file__), "data", "db")
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
    # Save the remote DB details as a JSON file in ../data/
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
    uvicorn.run(app, host="0.0.0.0", port=9000) 