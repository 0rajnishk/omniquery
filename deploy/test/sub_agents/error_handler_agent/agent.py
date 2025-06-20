from google.adk.agents import Agent

error_handler_agent = Agent(
    name="error_handler_agent",
    model="gemini-2.0-flash",
    description="Error handler agent for handling errors",
    instruction="I am error handler agent",
    sub_agents=[],
    tools=[],
) 