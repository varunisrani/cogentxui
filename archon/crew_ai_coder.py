from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os
import sys
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from openai import AsyncOpenAI
from supabase import Client
from utils.utils import get_env_var

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

@dataclass
class CrewAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    reasoner_output: str

system_prompt = """
~~ CONTEXT: ~~

You are an expert at CrewAI - a framework for orchestrating autonomous AI agents. You have access to all the CrewAI documentation,
including examples, API reference, and resources to help users build CrewAI agents and crews.

~~ GOAL: ~~

Your job is to help users create AI agents and crews with CrewAI.
The user will describe what they want to build, and you will help them create it using CrewAI's framework.
You will search through the CrewAI documentation with the provided tools to find all necessary information.

It's important to search through multiple CrewAI documentation pages to get complete information.
Use RAG and documentation tools multiple times when creating agents/crews from scratch.

~~ STRUCTURE: ~~

When building a CrewAI solution from scratch, split the code into these files:
- `agents.py`: Define all CrewAI agents with their roles, goals and tools
- `tasks.py`: Define the tasks that agents will work on
- `tools.py`: Custom tool functions used by the agents
- `crew.py`: Main file that creates and runs the crew with the agents and tasks
- `.env.example`: Environment variables needed with comments explaining each
- `requirements.txt`: Required package names (no versions)

~~ INSTRUCTIONS: ~~

- Take action without asking the user first. Always check documentation before writing code.
- Start with RAG search, then check available documentation pages for more details.
- Be honest when you can't find something in the docs.
- When building new crews/agents, provide complete working code.
- For refinements, only show the necessary code changes.
- Always ask users if they need changes or if the code looks good.
"""

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model=get_env_var('EMBEDDING_MODEL') or 'text-embedding-3-small',
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536

async def retrieve_relevant_documentation(ctx: CrewAIDeps, user_query: str) -> str:
    """Retrieve relevant documentation chunks based on the query with RAG."""
    try:
        query_embedding = await get_embedding(user_query, ctx.openai_client)
        
        result = ctx.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'crew_ai_docs'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

async def list_documentation_pages(ctx: CrewAIDeps) -> List[str]:
    """List all available CrewAI documentation pages."""
    try:
        result = ctx.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'crew_ai_docs') \
            .execute()
        
        if not result.data:
            return []
            
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error listing pages: {e}")
        return []

async def get_page_content(ctx: CrewAIDeps, url: str) -> str:
    """Get full content of a specific documentation page."""
    try:
        result = ctx.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'crew_ai_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        page_title = result.data[0]['title'].split(' - ')[0]
        formatted_content = [f"# {page_title}\n"]
        
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

# Create the CrewAI agent that helps build other agents/crews
crew_ai_expert = Agent(
    name="CrewAI Expert",
    role="AI Agent Framework Expert",
    goal="Help users create effective AI agents and crews using CrewAI",
    backstory="""You are an expert in creating AI agent teams with CrewAI. 
    You have deep knowledge of the framework and best practices for building autonomous AI systems.""",
    tools=[
        Tool(
            name="retrieve_docs",
            func=retrieve_relevant_documentation,
            description="Search CrewAI documentation for relevant information"
        ),
        Tool(
            name="list_pages",
            func=list_documentation_pages,
            description="List all available CrewAI documentation pages"
        ),
        Tool(
            name="get_page",
            func=get_page_content,
            description="Get the full content of a specific documentation page"
        )
    ]
) 