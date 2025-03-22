from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import asyncio
import json
import uuid
import logging
import time
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Set API key and model directly
os.environ["OPENAI_API_KEY"] = "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"
os.environ["LLM_API_KEY"] = "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"
os.environ["REASONER_MODEL"] = "gpt-4o-mini"
os.environ["PRIMARY_MODEL"] = "gpt-4o-mini"
os.environ["BASE_URL"] = "https://api.openai.com/v1"

# Import agent modules
try:
    from archon.archon_graph import agentic_flow
    from archon.pydantic_ai_coder import PydanticAIDeps
    AGENT_AVAILABLE = True
except ImportError:
    logging.warning("Agent modules could not be imported. Falling back to echo mode.")
    AGENT_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(thread)d] - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('api_server')

app = FastAPI(title="Agent Chat API")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

class UserMessage(BaseModel):
    message: str
    thread_id: Optional[str] = None

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"[HTTP] {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.2f}s")
    return response

@app.get("/")
async def root():
    return {"message": "Agent Chat API is running", "agent_available": AGENT_AVAILABLE}

@app.get("/api/confirm")
async def confirm_connection() -> Dict[str, bool]:
    logger.info("[Health] Health check request received")
    return {"connected": True}

@app.get("/api/health")
async def health_check():
    status = "healthy"
    components = {"api": True, "agent": AGENT_AVAILABLE}
    
    # Try to check agent health if available
    if AGENT_AVAILABLE:
        try:
            # Simple test configuration
            test_config = {"configurable": {"thread_id": "test"}}
            # Try a simple invoke to check if agent is working
            await agentic_flow.ainvoke({"latest_user_message": "test", "messages": []}, test_config)
            components["agent_responsive"] = True
        except Exception as e:
            logger.warning(f"Agent system check failed: {str(e)}")
            components["agent_responsive"] = False
            status = "degraded"
    
    return {
        "status": status,
        "components": components,
        "active_connections": len(active_connections),
        "timestamp": time.time()
    }

@app.post("/api/chat")
async def chat(user_message: UserMessage, request: Request):
    """
    Endpoint for non-streaming chat interactions
    """
    start_time = time.time()
    thread_id = user_message.thread_id or str(uuid.uuid4())
    try:
        logger.info(f"[HTTP] New chat request received - Thread ID: {thread_id}")
        logger.debug(f"[HTTP] Message content: {user_message.message}")
        
        # Process the message using agent system if available
        if AGENT_AVAILABLE:
            try:
                # Configure agent with thread ID
                config = {
                    "configurable": {
                        "thread_id": thread_id
                    }
                }
                
                # Invoke agent system
                result = await agentic_flow.ainvoke(
                    {
                        "latest_user_message": user_message.message,
                        "messages": []  # In a real app, you'd fetch previous messages
                    },
                    config
                )
                
                # Extract response text
                if isinstance(result, dict) and "scope" in result:
                    # If the result contains a scope, use that as the response
                    response_text = result.get("scope", "")
                else:
                    # Default response format
                    response_text = str(result) if result else "No response from agent"
                
            except Exception as e:
                logger.error(f"[HTTP] Error in agent processing: {str(e)}", exc_info=True)
                # Fall back to echo response
                response_text = f"Agent error: {str(e)}\n\nFallback: You said: {user_message.message}"
        else:
            # Echo mode
            response_text = f"You said: {user_message.message}\n\nThis is a response from the API server (no agent available)."
        
        process_time = time.time() - start_time
        logger.info(f"[HTTP] Message processed successfully - Thread ID: {thread_id} - Time: {process_time:.2f}s")
        
        return {
            "response": response_text,
            "thread_id": thread_id,
            "success": True,
            "is_first_message": user_message.thread_id is None
        }
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"[HTTP] Error processing chat request - Thread ID: {thread_id} - Time: {process_time:.2f}s", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat/{thread_id}")
async def websocket_chat(websocket: WebSocket, thread_id: str):
    """
    WebSocket endpoint for streaming chat interactions
    """
    connection_id = f"{thread_id[:8]}..."  # Truncated ID for logging
    connection_start_time = time.time()
    
    try:
        await websocket.accept()
        active_connections[thread_id] = websocket
        logger.info(f"[WS] New connection established - Thread ID: {connection_id}")
        logger.info(f"[WS] Active connections count: {len(active_connections)}")
        
        while True:
            message_start_time = time.time()
            try:
                data = await websocket.receive_text()
                message_data = json.loads(data)
                user_message = message_data.get("message", "")
                logger.info(f"[WS] Message received - Thread ID: {connection_id}")
                
                if not user_message.strip():
                    logger.warning(f"[WS] Empty message received - Thread ID: {connection_id}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": "Message cannot be empty"
                    }))
                    continue
                
                # Check if this is the first message or a continuation
                is_first_message = message_data.get("is_first_message", True)
                logger.info(f"[WS] Processing {'first' if is_first_message else 'continuation'} message - Thread ID: {connection_id}")
                
                if AGENT_AVAILABLE:
                    try:
                        # Set up agent configuration
                        config = {
                            "configurable": {
                                "thread_id": thread_id
                            }
                        }
                        
                        # Stream response from agent
                        chunk_count = 0
                        async for chunk in agentic_flow.astream(
                            {
                                "latest_user_message": user_message,
                                "messages": []  # Would store message history in real app
                            },
                            config,
                            stream_mode="custom"
                        ):
                            # Send each chunk to the client
                            chunk_count += 1
                            await websocket.send_text(json.dumps({
                                "type": "chunk",
                                "content": str(chunk)
                            }))
                            # Small delay to avoid overwhelming the client
                            await asyncio.sleep(0.01)
                        
                    except Exception as e:
                        logger.error(f"[WS] Agent streaming error: {str(e)}", exc_info=True)
                        # Fall back to simulated streaming
                        await websocket.send_text(json.dumps({
                            "type": "chunk",
                            "content": f"Agent error: {str(e)}\n\nFallback response: "
                        }))
                        # Then continue with echo fallback below
                        AGENT_AVAILABLE = False  # Temporarily disable for this session
                
                # Fallback or default mode: simulate streaming response
                if not AGENT_AVAILABLE:
                    # Simulate streaming response by sending chunks with delays
                    words = f"You said: {user_message}\n\nThis is a simulated streaming response from the API server. I'll send this response word by word to demonstrate streaming functionality."
                    words = words.split()
                    
                    chunk_count = 0
                    for word in words:
                        chunk = word + " "
                        chunk_count += 1
                        await websocket.send_text(json.dumps({
                            "type": "chunk",
                            "content": chunk
                        }))
                        await asyncio.sleep(0.1)  # Simulate thinking/processing time
                
                # Send completion message
                await websocket.send_text(json.dumps({
                    "type": "complete"
                }))
                message_time = time.time() - message_start_time
                logger.info(f"[WS] Message processing complete - Thread ID: {connection_id} - Total chunks: {chunk_count} - Time: {message_time:.2f}s")
                
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON received: {str(e)}"
                logger.error(f"[WS] {error_msg} - Thread ID: {connection_id}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": error_msg
                }))
                
    except WebSocketDisconnect:
        connection_time = time.time() - connection_start_time
        logger.info(f"[WS] Connection closed - Thread ID: {connection_id} - Duration: {connection_time:.2f}s")
        if thread_id in active_connections:
            del active_connections[thread_id]
            logger.info(f"[WS] Remaining active connections: {len(active_connections)}")
    except Exception as e:
        connection_time = time.time() - connection_start_time
        error_msg = f"WebSocket error: {str(e)}"
        logger.error(f"[WS] {error_msg} - Thread ID: {connection_id} - Duration: {connection_time:.2f}s", exc_info=True)
        if thread_id in active_connections:
            del active_connections[thread_id]
            try:
                await websocket.close()
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True) 
