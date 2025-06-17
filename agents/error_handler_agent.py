from datetime import datetime, timezone

from uuid import uuid4
import logging
import os
import json
import requests  # Added for ASI:ONE

from uagents import Agent, Protocol, Context
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
ASI_ONE_API_KEY = os.getenv('ASI_ONE_API_KEY')

if not GENAI_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your Google API key.")
if not ASI_ONE_API_KEY:
    raise ValueError("ASI_ONE_API_KEY environment variable not set. Please set it to your ASI:ONE API key.")

# Configure Gemini API
genai.configure(api_key=GENAI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# ASI:ONE API setup
ASI_ONE_URL = "https://api.asi1.ai/v1/chat/completions"
ASI_ONE_MODEL = "asi1-mini"  # or another model if desired
ASI_ONE_TEMPERATURE = 0.7  # Set your desired temperature here
ASI_ONE_MAX_TOKENS = 300  # Adjust as needed

error_handler = Agent(name="errorHandler", mailbox=True, port=9004, endpoint=["http://localhost:9004/submit"])
chat_proto = Protocol(spec=chat_protocol_spec)

@error_handler.on_event("startup")
async def startup_handler(ctx: Context):
    ctx.logger.info(f"My name is {ctx.agent.name} and my address is {ctx.agent.address}")

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    for item in msg.content:
        if isinstance(item, TextContent):
            try:
                request_id, user_query = item.text.split(":::", 1)
            except ValueError:
                ctx.logger.error(f"Malformed message format: {item.text}")
                continue
            ctx.logger.info(f"ErrorHandler received unclassified query from {sender}: {user_query} (Request ID: {request_id})")

            # Send acknowledgement
            ack = ChatAcknowledgement(
                timestamp=datetime.now(timezone.utc),
                acknowledged_msg_id=msg.msg_id
            )
            await ctx.send(sender, ack)

            # ASI:ONE prompt for fallback general knowledge answer
            prompt = f"""The following question could not be classified into any known type. Please use your own general knowledge and reasoning to answer it helpfully and clearly.\n\nUser Question: \"{user_query}\"\n\nAnswer:"""

            try:
                response_msg = asi_one_generate(prompt)
            except Exception as e:
                ctx.logger.error(f"ASI:ONE error in ErrorHandler: {e}")
                response_msg = "I couldn't process your question. Please try rephrasing it."

            reply = ChatMessage(
                timestamp=datetime.now(timezone.utc),
                msg_id=uuid4(),
                content=[TextContent(type="text", text=f"{request_id}:::{response_msg}")],
            )
            await ctx.send(sender, reply)

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for {msg.acknowledged_msg_id}")

error_handler.include(chat_proto, publish_manifest=True)

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
        logging.error(f"ASI:ONE API error: {e}")
        return "I'm unable to answer that right now."

if __name__ == "__main__":
    error_handler.run()
