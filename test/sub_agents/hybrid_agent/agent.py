from google.adk.agents import Agent

hybrid_agent = Agent(
    name="hybrid_agent",
    model="gemini-2.0-flash",
    description="Hybrid agent for combined operations",
    instruction="I am hybrid agent",
    sub_agents=[],
    tools=[],
) 