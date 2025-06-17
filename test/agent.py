from google.adk.agents import Agent

from .sub_agents.sql_agent.agent import sql_agent
from .sub_agents.document_agent.agent import document_agent
from .sub_agents.hybrid_agent.agent import hybrid_agent
from .sub_agents.error_handler_agent.agent import error_handler_agent

# Create the root test agent
test_agent = Agent(
    name="test_agent",
    model="gemini-2.0-flash",
    description="Test agent with multiple sub-agents",
    instruction="""
    You are the primary test agent that coordinates between different specialized agents.
    Your role is to help users with their queries and direct them to the appropriate specialized agent.

    You have access to the following specialized agents:

    1. SQL Agent
       - Asks for structured data like billing, test results, admissions, discharge dates, medications, etc.(Age,Gender,Blood Type,Medical Condition,Date of Admission,Doctor,Hospital,Insurance Provider,Billing Amount,Room Number,Admission Type	Discharge, Date, Medication, Test Results)

    2. Document Agent
       - background, biodata, history, doctor notes, symptoms observed, treatment plan, prescribed medications, follow-up recommendations, lifestyle adjustments, or other narrative details.

    3. Hybrid Agent
       - Requires both patient background and structured medical data

    4. Error Handler Agent
       - Vague, incomplete, or ambiguous queries

    Direct users to the appropriate agent based on their needs.
    """,
    sub_agents=[sql_agent, document_agent, hybrid_agent, error_handler_agent],
    tools=[],
) 