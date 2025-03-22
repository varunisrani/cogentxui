from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import json
from archon_graph import agentic_flow, AgentState
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from fastapi import WebSocketDisconnect
from fastapi.websockets import WebSocketState
import uuid

app = FastAPI(title="Archon AI API", version="1.0.0")

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your Next.js URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions and their websockets
active_sessions: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, WebSocket] = {}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    messages: List[ChatMessage]
    context: Dict[str, Any]

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    
    # Initialize or get session state
    if session_id not in active_sessions:
        active_sessions[session_id] = {
            'messages': [],
            'agent_state': {
                'messages': [],
                'scope': '',
                'architecture': '',
                'latest_user_message': ''
            }
        }
    
    session = active_sessions[session_id]
    
    # Update agent state with new message
    session['agent_state']['latest_user_message'] = request.message
    session['messages'].append(ChatMessage(role="user", content=request.message))
    
    try:
        # Create a response accumulator
        response_content = []
        
        def stream_handler(content: str):
            response_content.append(content)
        
        # Run the agent workflow
        result = await agentic_flow.ainvoke(
            session['agent_state'],
            config={"configurable": {"stream_handler": stream_handler}}
        )
        
        # Update session state with result
        session['agent_state'] = result
        
        # Combine response content
        full_response = ''.join(response_content)
        
        # Add assistant's response to messages
        session['messages'].append(ChatMessage(role="assistant", content=full_response))
        
        return ChatResponse(
            session_id=session_id,
            response=full_response,
            messages=session['messages'],
            context={
                'scope': session['agent_state'].get('scope', ''),
                'architecture': session['agent_state'].get('architecture', '')
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    try:
        await websocket.accept()
        
        # Store WebSocket connection
        websocket_connections[session_id] = websocket
        
        # Initialize or get session state
        if session_id not in active_sessions:
            active_sessions[session_id] = {
                'messages': [],
                'agent_state': {
                    'messages': [],
                    'scope': '',
                    'architecture': '',
                    'latest_user_message': '',
                    'thread_id': str(uuid.uuid4())  # Add thread_id
                }
            }
        
        session = active_sessions[session_id]
        
        try:
            while True:
                # Receive message from client
                message = await websocket.receive_text()
                data = json.loads(message)
                
                # Update agent state with new message
                session['agent_state']['latest_user_message'] = data['message']
                session['messages'].append({"role": "user", "content": data['message']})
                
                # Create a stream handler that sends chunks through websocket
                async def stream_handler(content: str):
                    if websocket.client_state == WebSocketState.CONNECTED:
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "chunk",
                                "content": content
                            }))
                        except Exception as e:
                            print(f"Error sending chunk: {e}")
                
                try:
                    # Run the agent workflow
                    result = await agentic_flow.ainvoke(
                        session['agent_state'],
                        config={"configurable": {"stream_handler": stream_handler}}
                    )
                    
                    # Update session state with result
                    session['agent_state'].update(result)
                    
                    # Send final state update
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text(json.dumps({
                            "type": "complete",
                            "context": {
                                'scope': session['agent_state'].get('scope', ''),
                                'architecture': session['agent_state'].get('architecture', '')
                            }
                        }))
                except Exception as e:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "error": str(e)
                        }))
                    print(f"Error in agent workflow: {e}")
                    
        except WebSocketDisconnect:
            print(f"WebSocket {session_id} disconnected normally")
        except Exception as e:
            print(f"Error in WebSocket connection {session_id}: {e}")
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
    except Exception as e:
        print(f"Error in WebSocket setup {session_id}: {e}")
    finally:
        # Clean up
        if session_id in websocket_connections:
            del websocket_connections[session_id]
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

@app.delete("/api/chat/{session_id}")
async def clear_chat(session_id: str):
    if session_id in active_sessions:
        active_sessions[session_id] = {
            'messages': [],
            'agent_state': {
                'messages': [],
                'scope': '',
                'architecture': '',
                'latest_user_message': ''
            }
        }
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 