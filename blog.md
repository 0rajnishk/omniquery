# OmniQuery: The End of Data Silos — A Story of Building Universal Analytics with Google ADK
#### Building OmniQuery: Universal Data Analysis with Google Agent Development Kit

*By Rajnish kumar — #adkhackathon

---

## Introduction

In the era of data-driven decision making, organizations are challenged by the diversity and fragmentation of their data sources. Structured databases, spreadsheets, and unstructured documents all hold valuable insights, but integrating and analyzing them together is a complex task. For the Google Cloud Agent Development Kit Hackathon, I set out to solve this challenge by building **OmniQuery** — an intelligent, multi-agent system that unifies data analysis across all formats, powered by the Agent Development Kit (ADK).

---

## The Vision: Universal Data Intelligence

The goal was simple but ambitious: create a platform where users can upload any data (PDFs, Word, Excel, CSV, or connect to databases), ask questions in plain English, and receive comprehensive, AI-powered insights that combine all available sources. No more data silos, no more manual cross-referencing — just one interface for all your analytics needs.

---

## Why Agent Development Kit?

Google's Agent Development Kit (ADK) provided the perfect foundation for this vision. With its modular, agent-based architecture, ADK made it easy to:
- Orchestrate specialized agents for different data types
- Route queries intelligently based on user intent
- Enable collaborative analysis between agents
- Scale and extend the system for new data sources

The ADK's seamless integration with Google Cloud services (BigQuery, Cloud Storage, Vertex AI) meant I could focus on building intelligence, not infrastructure.

---

## Architecture Overview

![System Flow Diagram](https://raw.githubusercontent.com/0rajnishk/omniquery/refs/heads/master/screenshots/Flowdiagram.png)
*High-level architecture and agent workflow for OmniQuery.*

- **Frontend**: FastAPI + Vue.js for a modern, interactive UI
- **Backend**: Multi-agent system using ADK, orchestrating:
  - **SQL Agent**: Handles BigQuery and database queries
  - **Document Agent**: Analyzes PDFs and Word docs with semantic search
  - **Hybrid Agent**: Synthesizes insights from both structured and unstructured data
  - **Error Handler Agent**: Provides helpful troubleshooting and fallback responses
- **Storage**: Google Cloud Storage for files, BigQuery for structured data
- **LLM Integration**: Vertex AI and Gemini for natural language understanding and summarization

---

## Building the Agents

### 1. SQL Agent
- Connects to BigQuery using the Python client
- Retrieves schema and executes queries on uploaded or connected datasets
- Converts natural language questions into safe, efficient SQL

### 2. Document Agent
- Uses FAISS and Gemini embeddings for semantic search over uploaded PDFs/Word files
- Finds and summarizes relevant information from unstructured documents

### 3. Hybrid Agent
- Combines results from both SQL and document agents
- Synthesizes a unified answer using LLMs, providing context-aware, cross-source insights

### 4. Error Handler Agent
- Catches and explains errors, offering troubleshooting steps and fallback suggestions

---

## Data Ingestion & User Experience

- **Upload**: Users can upload PDFs, Word, Excel, and CSV files, or connect to external databases
- **Auto-Processing**: Files are automatically indexed (documents) or imported to BigQuery (spreadsheets)
- **Query**: Users ask questions in plain English; the system routes and combines answers from all relevant agents
- **Presentation**: Results are returned as clear, actionable summaries, with the option for presentation-ready views

---

## Key Challenges & Solutions

- **Schema Unification**: Automatically mapping columns and relationships across files and databases
- **Safe SQL Generation**: Ensuring only safe, read-only queries are executed
- **Scalability**: Leveraging Google Cloud for storage and compute scalability
- **User Experience**: Designing a UI that makes complex analytics accessible to everyone

---

## Screenshots

![Chat Interface](https://raw.githubusercontent.com/0rajnishk/omniquery/refs/heads/master/screenshots/chat.png)
*Conversational interface for querying and analyzing data.*

![Database View](https://raw.githubusercontent.com/0rajnishk/omniquery/refs/heads/master/screenshots/db.png)
*View and manage structured data sources and query results.*

![Excel Upload](https://raw.githubusercontent.com/0rajnishk/omniquery/refs/heads/master/screenshots/excel.png)
*Upload and process Excel files for analysis.*

![Document Analysis](https://raw.githubusercontent.com/0rajnishk/omniquery/refs/heads/master/screenshots/document.png)
*Analyze and extract insights from PDF and Word documents.*

![Presentation Mode](https://raw.githubusercontent.com/0rajnishk/omniquery/refs/heads/master/screenshots/presentation.png)
*Generate and view presentation-ready insights and summaries.*

![Info Page](https://raw.githubusercontent.com/0rajnishk/omniquery/refs/heads/master/screenshots/info.png)
*Information and help page for users.*

---

## Lessons Learned

- The agent-based approach is incredibly powerful for modular, extensible analytics
- Google Cloud's managed services (BigQuery, Cloud Storage, Vertex AI) make it possible to build scalable, production-grade systems quickly
- User-centric design is key: hiding complexity and surfacing insights in natural language makes analytics accessible to everyone

---

## Try It Yourself!

- **Live Demo:** https://adk-trial-289215770101.asia-south1.run.app/
- **Login:**
  - Username: `admin`
  - Password: `123456`
- Upload your own data and start asking questions!

---

## Conclusion

OmniQuery demonstrates the power of combining multi-agent intelligence with cloud-native analytics. By breaking down data silos and enabling universal, conversational analysis, it paves the way for the next generation of business intelligence platforms.

*Built for the Google Cloud Agent Development Kit Hackathon — #adkhackathon* 