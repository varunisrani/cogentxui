from __future__ import annotations
from typing import Literal, TypedDict
from langgraph.types import Command
import os

import streamlit as st
import logfire
import asyncio
import time
import json
import uuid
import sys
import platform
import subprocess
import threading
import queue
import webbrowser
import importlib
from urllib.parse import urlparse
from openai import AsyncOpenAI
from supabase import Client, create_client
from dotenv import load_dotenv
from utils.utils import get_env_var, save_env_var, write_to_log
from future_enhancements import future_enhancements_tab
from archon.archon_graph import agentic_flow
import httpx

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Initialize clients
openai_client = None
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'
is_ollama = "localhost" in base_url.lower()

if is_ollama:
    openai_client = AsyncOpenAI(base_url=base_url,api_key=api_key)
elif get_env_var("OPENAI_API_KEY"):
    openai_client = AsyncOpenAI(api_key=get_env_var("OPENAI_API_KEY"))
else:
    openai_client = None

if get_env_var("SUPABASE_URL"):
    supabase: Client = Client(
            get_env_var("SUPABASE_URL"),
            get_env_var("SUPABASE_SERVICE_KEY")
        )
else:
    supabase = None

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="CogentX - Agent Builder",
    page_icon="🤖",
    layout="wide",
)

# Set custom theme colors to match CogentX logo (green and pink)
# Primary color (green) and secondary color (pink)
st.markdown("""
    <style>
    :root {
        --primary-color: #00CC99;  /* Green */
        --secondary-color: #EB2D8C; /* Pink */
        --text-color: #262730;
    }
    
    /* Style the buttons */
    .stButton > button {
        color: white;
        border: 2px solid var(--primary-color);
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        color: white;
        border: 2px solid var(--secondary-color);
    }
    
    /* Override Streamlit's default focus styles that make buttons red */
    .stButton > button:focus, 
    .stButton > button:focus:hover, 
    .stButton > button:active, 
    .stButton > button:active:hover {
        color: white !important;
        border: 2px solid var(--secondary-color) !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Style headers */
    h1, h2, h3 {
        color: var(--primary-color);
    }
    
    /* Hide spans within h3 elements */
    h1 span, h2 span, h3 span {
        display: none !important;
        visibility: hidden;
        width: 0;
        height: 0;
        opacity: 0;
        position: absolute;
        overflow: hidden;
    }
    
    /* Style code blocks */
    pre {
        border-left: 4px solid var(--primary-color);
    }
    
    /* Style links */
    a {
        color: var(--secondary-color);
    }
    
    /* Style the chat messages */
    .stChatMessage {
        border-left: 4px solid var(--secondary-color);
    }
    
    /* Style the chat input */
    .stChatInput > div {
        border: 2px solid var(--primary-color) !important;
    }
    
    /* Remove red outline on focus */
    .stChatInput > div:focus-within {
        box-shadow: none !important;
        border: 2px solid var(--secondary-color) !important;
        outline: none !important;
    }
    
    /* Remove red outline on all inputs when focused */
    input:focus, textarea:focus, [contenteditable]:focus {
        box-shadow: none !important;
        border-color: var(--secondary-color) !important;
        outline: none !important;
    }

    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_thread_id():
    return str(uuid.uuid4())

thread_id = get_thread_id()

async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        # First message from user
        if len(st.session_state.messages) == 1:
            async for msg in agentic_flow.astream(
                    {"latest_user_message": user_input}, config, stream_mode="custom"
                ):
                    yield msg
        # Continue the conversation
        else:
            async for msg in agentic_flow.astream(
                Command(resume=user_input), config, stream_mode="custom"
            ):
                yield msg
    except httpx.RemoteProtocolError as e:
        # Handle connection errors gracefully
        error_message = "Connection interrupted. Please try again."
        yield error_message
    except Exception as e:
        # Handle other errors
        error_message = f"An error occurred: {str(e)}"
        yield error_message

async def chat_tab():
    """Display the chat interface for talking to CogentX"""

    st.write("Create an agent ready to prompts.")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        message_type = message["type"]
        if message_type in ["human", "ai", "system"]:
            with st.chat_message(message_type):
                st.markdown(message["content"])    

    # Chat input for the user
    user_input = st.chat_input("What do you want to build today?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append({"type": "human", "content": user_input})
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display assistant response in chat message container
        response_content = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # Placeholder for updating the message
            try:
                # Run the async generator to fetch responses
                async for chunk in run_agent_with_streaming(user_input):
                    if chunk.startswith("An error occurred") or chunk.startswith("Connection interrupted"):
                        message_placeholder.error(chunk)
                        return
                    response_content += chunk
                    # Update the placeholder with the current response content
                    message_placeholder.markdown(response_content)
                
                # Only append successful responses to message history
                st.session_state.messages.append({"type": "ai", "content": response_content})
            except Exception as e:
                error_message = f"An unexpected error occurred: {str(e)}"
                message_placeholder.error(error_message)

async def main():
    # Check for tab query parameter
    query_params = st.query_params
    if "tab" in query_params:
        tab_name = query_params["tab"]
        if tab_name in ["Chat"]:
            st.session_state.selected_tab = tab_name

    # Add sidebar navigation
    with st.sidebar:
        
        
        # Navigation options with vertical buttons
        st.write("### Navigation")
        
        # Initialize session state for selected tab if not present
        if "selected_tab" not in st.session_state:
            st.session_state.selected_tab = "Chat"
        
        # Vertical navigation buttons
        chat_button = st.button("Chat", use_container_width=True, key="chat_button")
        
        # Update selected tab based on button clicks
        if chat_button:
            st.session_state.selected_tab = "Chat"
    
    # Display the selected tab
    if st.session_state.selected_tab == "Chat":
        st.title("CogentX - Agent Builder")
        await chat_tab()

if __name__ == "__main__":
    asyncio.run(main())