from typing import TypedDict, List, Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import json
from pathlib import Path

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    try:
        from langgraph.graph import Graph
        StateGraph = Graph
        END = "__end__"
    except:
        raise ImportError("Please install langgraph: pip install langgraph>=0.2.0")

from src.backend.generation_layer import ChatCompletion
from src.artifacts import SystemInstruction
from src.backend.intent_moderation_check import IntentCheck, ModerationCheck
from src.database.chromadb_vectorstore import VectorStore
from src.backend.reranker import Reranker


# Initialize components
mod_chk = ModerationCheck()
int_cnf = IntentCheck()
chat_completion = ChatCompletion()
vector_store = VectorStore()
reranker = Reranker()

app = FastAPI(title="Mr Help Mate AI", version="1.0.0")


# Define graph state
class State(TypedDict, total=False):
    messages: List[Dict[str, str]]
    input_message: str
    ai_response: Optional[str]
    input_flagged: bool
    intent_ok: bool
    ai_flagged: bool
    documents: List[str]
    reranked_context: str
    is_greeting: bool


def is_greeting_message(message: str) -> bool:
    """Check if message is a greeting"""
    greetings = [
        "hello", "hi", "hey", "greetings", "good morning", 
        "good afternoon", "good evening", "howdy", "what's up",
        "whats up", "sup", "yo", "hiya", "heya"
    ]
    message_lower = message.lower().strip()
    
    # Check if message is short and contains greeting words
    if len(message_lower.split()) <= 5:
        for greeting in greetings:
            if greeting in message_lower:
                return True
    return False


def generate_greeting_response(message: str) -> str:
    """Generate a simple greeting response"""
    greetings_map = {
        "good morning": "Good morning! How can I assist you with insurance today?",
        "good afternoon": "Good afternoon! How can I assist you with insurance today?",
        "good evening": "Good evening! How can I assist you with insurance today?",
    }
    
    message_lower = message.lower().strip()
    
    for key, response in greetings_map.items():
        if key in message_lower:
            return response
    
    return "Hello! How can I assist you with insurance today?"


# ==================== Graph Nodes ====================
def check_greeting(state: State) -> dict:
    """Check if input is a greeting"""
    is_greet = is_greeting_message(state["input_message"])
    if is_greet:
        response = generate_greeting_response(state["input_message"])
        return {"is_greeting": True, "ai_response": response}
    return {"is_greeting": False}


def check_input_moderation(state: State) -> dict:
    """Check if input message is flagged"""
    try:
        flagged = mod_chk.check_moderation(input_message=state["input_message"])
        return {"input_flagged": flagged}
    except Exception as e:
        print(f"Moderation check error: {e}")
        return {"input_flagged": False}


def intent_check_node(state: State) -> dict:
    """Check if intent is valid"""
    try:
        ok = int_cnf.check_intent(input_message=state["input_message"])
        return {"intent_ok": ok}
    except Exception as e:
        print(f"Intent check error: {e}")
        return {"intent_ok": True}  # Allow by default on error


def out_of_scope_node(state: State) -> dict:
    """Handle out of scope queries"""
    response = "Your question is out of scope or context. Please ask a question related to insurance."
    return {"ai_response": response}


def retrieve_node(state: State) -> dict:
    """Retrieve relevant documents"""
    try:
        top_10 = vector_store.query_from_db(query=state["input_message"], top_k=10)
        return {"documents": top_10.get("documents", [])}
    except Exception as e:
        print(f"Retrieval error: {e}")
        return {"documents": []}


def rerank_node(state: State) -> dict:
    """Rerank retrieved documents"""
    try:
        pairs = reranker.rerank_documents(
            documents=state["documents"],
            query=state["input_message"],
            top_k=3,
        )
        ctx = ""
        for score, doc in pairs:
            ctx += doc + "\n\n\n"
        return {"reranked_context": ctx}
    except Exception as e:
        print(f"Reranking error: {e}")
        return {"reranked_context": ""}


def chat_node(state: State) -> dict:
    """Generate chat response"""
    try:
        # Append user message with context
        new_msgs = list(state["messages"]) + [
            {
                "role": "user",
                "content": state["input_message"] + "\n\n###context:\n" + state.get("reranked_context", ""),
            }
        ]
        ai = chat_completion.chat_completion(messages=new_msgs)
        return {"messages": new_msgs, "ai_response": ai}
    except Exception as e:
        print(f"Chat generation error: {e}")
        return {"ai_response": "I apologize, but I encountered an error processing your request."}


def check_ai_moderation(state: State) -> dict:
    """Check if AI response is flagged"""
    try:
        flagged = mod_chk.check_moderation(input_message=state["ai_response"])
        return {"ai_flagged": flagged}
    except Exception as e:
        print(f"AI moderation check error: {e}")
        return {"ai_flagged": False}


def flagged_node(state: State) -> dict:
    """Handle flagged content"""
    response = "Your conversation has been flagged. Please maintain appropriate communication."
    return {"ai_response": response}


def respond_node(state: State) -> dict:
    """Final response node"""
    return {}


# ==================== Build Graph ====================
def build_chatbot_graph():
    """Build and compile the chatbot graph"""
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("check_greeting", check_greeting)
    graph.add_node("check_input_moderation", check_input_moderation)
    graph.add_node("intent_check", intent_check_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("chat", chat_node)
    graph.add_node("check_ai_moderation", check_ai_moderation)
    graph.add_node("flagged", flagged_node)
    graph.add_node("respond", respond_node)

    # Set entry point
    graph.set_entry_point("check_greeting")

    # Add conditional edges
    graph.add_conditional_edges(
        "check_greeting",
        lambda s: "respond" if s.get("is_greeting") else "check_input_moderation",
        {"respond": "respond", "check_input_moderation": "check_input_moderation"},
    )

    graph.add_conditional_edges(
        "check_input_moderation",
        lambda s: "flagged" if s.get("input_flagged") else "intent_check",
        {"flagged": "flagged", "intent_check": "intent_check"},
    )

    graph.add_conditional_edges(
        "intent_check",
        lambda s: "retrieve" if s.get("intent_ok") else "out_of_scope",
        {"retrieve": "retrieve", "out_of_scope": "out_of_scope"},
    )

    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "chat")
    graph.add_edge("chat", "check_ai_moderation")

    graph.add_conditional_edges(
        "check_ai_moderation",
        lambda s: "flagged" if s.get("ai_flagged") else "respond",
        {"flagged": "flagged", "respond": "respond"},
    )

    # End nodes
    graph.add_edge("out_of_scope", END)
    graph.add_edge("flagged", END)
    graph.add_edge("respond", END)

    return graph.compile()


# Compile the graph
chatbot_graph = build_chatbot_graph()

# Store conversation sessions
sessions: Dict[str, List[Dict[str, str]]] = {}


# ==================== FastAPI Routes ====================
@app.get("/")
async def serve_index():
    """Serve the index.html file"""
    index_path = Path("index.html")
    if not index_path.exists():
        return {"error": "index.html not found. Please ensure it's in the same directory as main.py"}
    return FileResponse("index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Mr Help Mate AI",
        "version": "1.0.0"
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    # Initialize session if not exists
    if session_id not in sessions:
        sessions[session_id] = [
            {"role": "system", "content": SystemInstruction.prompt}
        ]
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "").strip()
            
            if not user_message:
                continue
            
            try:
                # Run the graph with the user message
                final_state = chatbot_graph.invoke(
                    {
                        "messages": sessions[session_id],
                        "input_message": user_message,
                    }
                )
                
                # Get the AI response
                ai_response = final_state.get("ai_response", "I apologize, but I couldn't process your request.")
                
                # Update session messages only if not a greeting
                if not final_state.get("is_greeting", False):
                    sessions[session_id] = final_state.get("messages", sessions[session_id])
                
                # Send response back to client
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "message": ai_response
                }))
                
            except Exception as e:
                print(f"Error processing message: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "An error occurred while processing your request. Please try again."
                }))
    
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
        # Clean up old sessions
        if session_id in sessions:
            del sessions[session_id]
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn
    print("Starting Mr Help Mate AI server...")
    print("Server will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")