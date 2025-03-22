from __future__ import annotations as _annotations

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, RunContext
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List, Any
from langgraph.config import get_stream_writer
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import Client
import logfire
import os
import sys
import logging

# Import the message classes from Pydantic AI
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter
)

# Add the parent directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from archon.pydantic_ai_coder import (
    pydantic_ai_coder,
    PydanticAIDeps,
    list_documentation_pages_helper,
    implementation_agent,
    find_similar_agent_templates,
    adapt_template_code
)
from archon.utils.utils import get_env_var

# Load environment variables
load_dotenv()

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('agent_workflow')

base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'

is_ollama = "localhost" in base_url.lower()
is_anthropic = "anthropic" in base_url.lower()
is_openai = "openai" in base_url.lower()

reasoner_llm_model_name = get_env_var('REASONER_MODEL') or 'o3-mini'
reasoner_llm_model = AnthropicModel(reasoner_llm_model_name, api_key=api_key) if is_anthropic else OpenAIModel(reasoner_llm_model_name, base_url=base_url, api_key=api_key)

reasoner = Agent(  
    reasoner_llm_model,
    system_prompt="""You are an expert at defining scope and requirements for multi-agent CrewAI systems.
    
Your core responsibilities:

1. Requirements Analysis:
   - Understand user's needs for AI agent/crew creation
   - Break down requirements into distinct agent roles (2-4 agents)
   - Identify key functionalities for each agent
   - Map required tools and integrations per agent

2. Agent Role Definition:
   - Define 2-4 specialized agent roles
   - Ensure complementary capabilities
   - Avoid single-agent solutions
   - Design effective collaboration patterns

3. Documentation Research:
   - Search CrewAI documentation thoroughly
   - Identify relevant examples and patterns
   - Find appropriate tools for each agent
   - Document any gaps in documentation

4. Scope Definition:
   - Create detailed project scope
   - Define each agent's responsibilities
   - Outline inter-agent workflows
   - Specify success criteria per agent

5. Architecture Planning:
   - Design multi-agent structure
   - Plan agent interactions
   - Configure tool distribution
   - Ensure scalable communication

Always create comprehensive scope documents that include:
1. Multi-agent architecture diagram
2. Agent roles and responsibilities
3. Inter-agent communication patterns
4. Tool distribution across agents
5. Data flow between agents
6. Testing strategy per agent
7. Relevant documentation references

Your scope documents should enable implementation of effective multi-agent solutions."""
)

# Add this line to define primary_llm_model before its usage
primary_llm_model_name = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
primary_llm_model = AnthropicModel(primary_llm_model_name, api_key=api_key) if is_anthropic else OpenAIModel(primary_llm_model_name, base_url=base_url, api_key=api_key)

# Now you can use primary_llm_model in the Agent instantiation
architecture_agent = Agent(  
    primary_llm_model,
    system_prompt='You are an expert software architect who designs robust and scalable systems. You analyze requirements and create detailed technical architectures.'
)

implementation_agent = Agent(  
    primary_llm_model,
    system_prompt='Implementation Planning Agent creates detailed technical specifications.'
)

coder_agent = Agent(  
    primary_llm_model,
    system_prompt='Code Implementation Agent implements the solution.'
)

router_agent = Agent(  
    primary_llm_model,
    system_prompt="""You are an expert at understanding and routing user requests in a CrewAI development workflow, even when messages contain typos or are unclear.
    
Your core responsibilities:

1. Message Understanding:
   - Parse user intent even with typos/unclear wording
   - Extract key meaning from malformed messages
   - Handle multilingual input gracefully
   - Identify core request type regardless of format

2. Request Analysis:
   - Understand user's message intent
   - Identify if it's a new request or continuation
   - Determine if conversation should end
   - Route to appropriate next step

3. Conversation Flow:
   - Maintain context between messages
   - Track implementation progress
   - Handle edge cases gracefully
   - Adapt to user's communication style

4. Quality Control:
   - Verify message understanding
   - Route unclear messages for clarification
   - Ensure proper handling of all input types
   - Validate routing decisions

For each user message, analyze the intent and route to:
1. "general_conversation" - For general questions or chat
2. "create_agent" - For requests to create new agents/crews
3. "modify_code" - For requests to edit/update existing code
4. "unclear_input" - For messages needing clarification
5. "end_conversation" - For requests to end the conversation

Always focus on understanding the core intent, even if the message contains typos or is unclear."""
)

end_conversation_agent = Agent(  
    primary_llm_model,
    system_prompt="""You are an expert at providing final instructions for CrewAI agent setup and usage.
    
Your core responsibilities:

1. Setup Instructions:
   - Explain file organization
   - Detail environment setup
   - List required dependencies
   - Provide configuration steps

2. Usage Guide:
   - Show how to run the crew
   - Explain agent interactions
   - Demonstrate task execution
   - Provide example commands

3. Troubleshooting:
   - Common issues and solutions
   - Environment variables
   - Dependency conflicts
   - Error messages

4. Next Steps:
   - Testing recommendations
   - Monitoring suggestions
   - Performance optimization
   - Future enhancements

For each conversation end:
1. Summarize what was created
2. List setup steps in order
3. Show example usage
4. Provide friendly goodbye

Always ensure users have everything they need to run their CrewAI solution."""
)

openai_client=None

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

# Define state schema
class AgentState(TypedDict):
    latest_user_message: str
    messages: Annotated[List[bytes], lambda x, y: x + y]
    scope: str
    architecture: str

# Scope Definition Node with Reasoner LLM
async def define_scope_with_reasoner(state: AgentState):
    # First, get the documentation pages so the reasoner can decide which ones are necessary
    documentation_pages = await list_documentation_pages_helper(supabase)
    documentation_pages_str = "\n".join(documentation_pages)

    # Then, use the reasoner to define the scope
    prompt = f"""
    User AI Agent Request: {state['latest_user_message']}
    
    Create detailed scope document for the AI agent including:
    - Architecture diagram
    - Core components
    - External dependencies
    - Testing strategy

    Also based on these documentation pages available:

    {documentation_pages_str}

    Include a list of documentation pages that are relevant to creating this agent for the user in the scope document.
    """

    result = await reasoner.run(prompt)
    scope = result.data

    # Get the directory one level up from the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    scope_path = os.path.join(parent_dir, "workbench", "scope.md")
    os.makedirs(os.path.join(parent_dir, "workbench"), exist_ok=True)

    with open(scope_path, "w", encoding="utf-8") as f:
        f.write(scope)
    
    return {"scope": scope}

# Architecture Node with Architecture Agent
async def create_architecture(state: AgentState):
    """Creates detailed technical architecture based on scope."""
    
    # Ensure that 'architecture' is initialized in the state
    if 'architecture' not in state:
        state['architecture'] = ""  # Initialize it to an empty string or appropriate default

    prompt = f"""
    Based on the following scope document:
    {state['scope']}
    
    Create a detailed technical architecture including:
    1. System components and their interactions
    2. Data flow between components
    3. API specifications and endpoints
    4. Technology stack recommendations
    5. Integration points with external systems
    6. Security considerations
    7. Scalability and performance design
    8. Deployment architecture
    """
    
    result = await architecture_agent.run(prompt)
    architecture_plan = result.data
    
    # Save architecture to file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    architecture_path = os.path.join(parent_dir, "workbench", "architecture.md")
    
    with open(architecture_path, "w", encoding="utf-8") as f:
        f.write(architecture_plan)
    
    # Ensure the architecture is set in the state
    state['architecture'] = architecture_plan  # Set the architecture in the state

    return {"architecture": architecture_plan}

# Implementation Plan Node with Implementation Agent
async def create_implementation_plan(state: AgentState):
    """Creates a detailed implementation plan based on the architecture."""
    
    prompt = f"""
    Based on the following architecture document:
    {state['architecture']}
    
    Create a detailed implementation plan including:
    1. Step-by-step implementation guide
    2. Required tools and libraries
    3. Code structure and organization
    4. Testing and validation strategies
    5. Deployment instructions
    """
    
    result = await implementation_agent.run(prompt)
    implementation_plan = result.data
    
    # Save implementation plan to file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    implementation_path = os.path.join(parent_dir, "workbench", "implementation_plan.md")
    
    with open(implementation_path, "w", encoding="utf-8") as f:
        f.write(implementation_plan)
    
    return {"implementation_plan": implementation_plan}

# Coding Node with Feedback Handling
async def coder_agent(state: AgentState, writer):    
    logger.info("Starting coder agent execution")
    logger.info(f"User message: {state['latest_user_message'][:100]}...")
    
    deps = PydanticAIDeps(
        supabase=supabase,
        openai_client=openai_client,
        reasoner_output=state['scope'],
        architecture_plan=state['architecture']
    )
    logger.info("Dependencies initialized")

    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))
    logger.info(f"Loaded {len(message_history)} messages from history")

    # First, try to find similar templates
    logger.info("Searching for similar templates...")
    try:
        # Create RunContext with all required arguments
        context = RunContext(
            deps=deps,
            model=pydantic_ai_coder.model,
            usage={},  # Initialize empty usage stats
            prompt=state['latest_user_message']  # Use the user's message as the prompt
        )
        
        purpose, template_code = await find_similar_agent_templates(context, state['latest_user_message'])
        
        if template_code and isinstance(template_code, dict) and any(template_code.values()):
            logger.info(f"Found matching template with purpose: {purpose}")
            writer(f"\nFound similar template: {purpose}\nAdapting template to your requirements...\n")
            
            # Create a new context for template adaptation
            adapt_context = RunContext(
                deps=deps,
                model=pydantic_ai_coder.model,
                usage={},
                prompt=f"Adapt template for: {state['latest_user_message']}"
            )
            
            # Adapt the template code
            adapted_code = await adapt_template_code(adapt_context, template_code, state['latest_user_message'])
            
            if adapted_code and isinstance(adapted_code, dict) and any(adapted_code.values()):
                logger.info("Successfully adapted template code")
                # Create the files from adapted code
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                output_dir = os.path.join(parent_dir, "workbench")
                os.makedirs(output_dir, exist_ok=True)
                
                for file_type, code in adapted_code.items():
                    if not isinstance(code, str) or not code.strip():
                        logger.warning(f"Skipping invalid code for {file_type}")
                        continue
                        
                    file_name = file_type.replace('_code', '.py')
                    file_path = os.path.join(output_dir, file_name)
                    logger.info(f"Writing adapted code to {file_path}")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(code)
                    writer(f"\nCreated {file_name}")
                
                writer("\nTemplate successfully adapted and files created!")
                return {"messages": []}
            else:
                logger.warning("Template adaptation failed or returned invalid format")
        else:
            logger.info("No valid template code found, proceeding with standard code generation")
    except Exception as e:
        logger.error(f"Error in template processing: {str(e)}", exc_info=True)
        logger.info("Proceeding with standard code generation due to template processing error")
    
    # Run the agent in a stream for any additional modifications or if no template was found
    logger.info("Starting code generation/modification stream")
    if not is_openai:
        writer = get_stream_writer()
        result = await pydantic_ai_coder.run(state['latest_user_message'], deps=deps, message_history=message_history)
        writer(result.data)
        logger.info("Completed non-OpenAI code generation")
    else:
        async with pydantic_ai_coder.run_stream(
            state['latest_user_message'],
            deps=deps,
            message_history=message_history
        ) as result:
            async for chunk in result.stream_text(delta=True):
                writer(chunk)
        logger.info("Completed OpenAI code generation stream")

    logger.info("Coder agent execution completed")
    return {"messages": [result.new_messages_json()]}

# Interrupt the graph to get the user's next message
def get_next_user_message(state: AgentState):
    try:
        value = interrupt("What would you like to do next?")
        return {"latest_user_message": value}
    except Exception as e:
        if isinstance(e, GraphInterrupt):
            raise e
        return {"latest_user_message": str(e)}

# Determine if the user is finished creating their AI agent or not
async def route_user_message(state: AgentState):
    prompt = f"""
    The user has sent a message: 
    
    {state['latest_user_message']}

    If the user wants to end the conversation, respond with just the text "finish_conversation".
    If the user wants to continue coding the AI agent, respond with just the text "coder_agent".
    """

    result = await router_agent.run(prompt)
    
    if result.data == "finish_conversation": return "finish_conversation"
    return "coder_agent"

# End of conversation agent to give instructions for executing the agent
async def finish_conversation(state: AgentState, writer):    
    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    # Run the agent in a stream
    if not is_openai:
        writer = get_stream_writer()
        result = await end_conversation_agent.run(state['latest_user_message'], message_history= message_history)
        writer(result.data)   
    else: 
        async with end_conversation_agent.run_stream(
            state['latest_user_message'],
            message_history= message_history
        ) as result:
            # Stream partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                writer(chunk)

    return {"messages": [result.new_messages_json()]}        

# Build workflow
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("define_scope_with_reasoner", define_scope_with_reasoner)
builder.add_node("create_architecture", create_architecture)
builder.add_node("create_implementation_plan", create_implementation_plan)
builder.add_node("coder_agent", coder_agent)
builder.add_node("get_next_user_message", get_next_user_message)
builder.add_node("finish_conversation", finish_conversation)

# Set edges
builder.add_edge(START, "define_scope_with_reasoner")
builder.add_edge("define_scope_with_reasoner", "create_architecture")
builder.add_edge("create_architecture", "create_implementation_plan")
builder.add_edge("create_implementation_plan", "coder_agent")
builder.add_edge("coder_agent", "get_next_user_message")
builder.add_conditional_edges(
    "get_next_user_message",
    route_user_message,
    {"coder_agent": "coder_agent", "finish_conversation": "finish_conversation"}
)
builder.add_edge("finish_conversation", END)

# Configure persistence
memory = MemorySaver()
agentic_flow = builder.compile(checkpointer=memory)