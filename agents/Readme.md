# Agent Documentation

This system comprises a modular architecture where a central **Intent Classifier Agent** delegates tasks to several **worker agents** based on the nature of a user query. Each agent is built using the `uAgents` framework and communicates using a standardized **Chat Protocol**. Responses are generated using the **asi:one LLM API**.

---

## Intent Classifier Agent (Main Agent)

The **Intent Classifier Agent** serves as the central coordinator for processing incoming queries. It continuously monitors a local file (`query.txt`) for new user inputs and forwards those inputs to itself using the `ChatMessage` structure defined by the chat protocol. When it receives a message (either from a user or another agent), it uses the **asi:one LLM API** to classify the query into one of four categories: `sql`, `document`, `hybrid`, or `other`. Based on the classification, it forwards the original query to the appropriate worker agent (SQL Agent, Document Agent, Hybrid Agent, or Error Handler Agent). After the worker agent processes the query and replies, the Intent Classifier relays the response back to the original sender. It also logs acknowledgments and writes the final output to `response.txt`. This agent registers all worker agents locally via `ctx.register()` to ensure communication remains entirely within the local environment, avoiding remote endpoints.

---

## Document Agent (Worker Agent)

The **Document Agent** is responsible for handling queries that are best answered using information retrieval techniques over a vector-based document database. Upon receiving a message, it performs a similarity search against a locally stored FAISS vector index. The top relevant document passages retrieved from this search are then passed, along with the user's query, to the **asi:one LLM API**, which generates a context-aware answer. The response is then packaged into a `ChatMessage` and sent back to the sender (typically the Intent Classifier Agent). This agent operates entirely offline and ensures low-latency document querying by keeping the FAISS index pre-loaded in memory.

---

## SQL Agent (Worker Agent)

The **SQL Agent** handles all queries that require structured data retrieval from a relational database. When it receives a query message, it uses the **asi:one LLM API** to convert the natural language query into a syntactically correct SQL command, tailored to the database schema (which is either preloaded or fetched from a local schema file). This SQL command is executed against a local database using SQLAlchemy or a similar database connector. The resulting data, typically in tabular format, is then passed back into the LLM to be converted into a human-readable answer. The final formatted response is returned to the sender using a `ChatMessage`. This agent handles both SQL generation and execution internally and confirms receipt of every message with an acknowledgment.

---

## Hybrid Agent (Worker Agent)

The **Hybrid Agent** is designed for complex queries that require both structured data from a SQL database and unstructured information from document sources. Upon receiving a query, it internally invokes the logic of both the SQL Agent and Document Agent, generating a SQL query and retrieving relevant document vectors in parallel. The results from both sources are combined and sent to the **asi:one LLM API**, which synthesizes the final answer by integrating facts and context from both data modalities. This answer is then sent back to the sender as a `ChatMessage`. Unlike the Intent Classifier Agent, the Hybrid Agent does not send inter-agent messages; instead,


## Error Handler Agent (Worker Agent)

The **Error Handler Agent** serves as a fallback mechanism for handling unclassified or ambiguous queries. If the Intent Classifier is unable to determine a valid intent for a query (i.e., it falls into the "other" category), it forwards the message to this agent. The Error Handler responds with a predefined message indicating that the system could not understand or process the request. It does not use the LLM API and is intentionally kept lightweight to ensure fast failure handling. This guarantees that every user query, even those not fully supported, receives a graceful and informative response.

---