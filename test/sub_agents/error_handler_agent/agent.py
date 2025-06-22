from google.adk.agents import Agent

def error_reply(error_message: str = None) -> str:
    """
    Always return a helpful, friendly, and informative error message using the agent's own knowledge.
    """
    if not error_message:
        return (
            "I'm the error handler agent. It seems something went wrong. "
            "Here are some general troubleshooting steps:\n"
            "- Check your input for typos or missing information.\n"
            "- Make sure all required data is available.\n"
            "- If this is a system error, try again later or contact support.\n"
            "If you provide more details, I can offer more specific help!"
        )
    return (
        f"I'm the error handler agent. An error occurred: {error_message}\n"
        "Here are some things you can try:\n"
        "- Double-check your input.\n"
        "- Ensure all required fields are filled.\n"
        "- If this is a technical issue, please try again later.\n"
        "Let me know if you need more assistance!"
    )

error_handler_agent = Agent(
    name="error_handler_agent",
    model="gemini-2.0-flash",
    description="Error handler agent for handling errors",
    instruction="I am error handler agent. I provide helpful troubleshooting and error explanations.",
    sub_agents=[],
    tools=[error_reply],
) 