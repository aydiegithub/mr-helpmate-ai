from typing import TypedDict, List, Dict, Optional, Any

from langgraph.graph import StateGraph, END

from src.backend.generation_layer import ChatCompletion
from src.artifacts import SystemInstruction
from src.backend.intent_moderation_check import IntentCheck, ModerationCheck
from src.database.chromadb_vectorstore import VectorStore
from src.backend.reranker import Reranker


mod_chk = ModerationCheck()
int_cnf = IntentCheck()
chat_completion = ChatCompletion()
vector_store = VectorStore()
reranker = Reranker()


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


# Nodes
def check_input_moderation(state: State) -> dict:
    flagged = mod_chk.check_moderation(input_message=state["input_message"])
    return {"input_flagged": flagged}


def intent_check_node(state: State) -> dict:
    ok = int_cnf.check_intent(input_message=state["input_message"])
    return {"intent_ok": ok}


def out_of_scope_node(state: State) -> dict:
    print("Your question is out of scope or context, please ask the right question related to the domain.")
    return {}


def retrieve_node(state: State) -> dict:
    top_10 = vector_store.query_from_db(query=state["input_message"], top_k=10)
    return {"documents": top_10["documents"]}


def rerank_node(state: State) -> dict:
    pairs = reranker.rerank_documents(
        documents=state["documents"],
        query=state["input_message"],
        top_k=3,
    )
    ctx = ""
    for score, doc in pairs:
        ctx += doc + "\n\n\n"
    return {"reranked_context": ctx}


def chat_node(state: State) -> dict:
    # Append user message with context
    new_msgs = list(state["messages"]) + [
        {
            "role": "user",
            "content": state["input_message"] + "###context:\n" + state["reranked_context"],
        }
    ]
    ai = chat_completion.chat_completion(messages=new_msgs)
    return {"messages": new_msgs, "ai_response": ai}


def check_ai_moderation(state: State) -> dict:
    flagged = mod_chk.check_moderation(input_message=state["ai_response"])
    return {"ai_flagged": flagged}


def flagged_node(state: State) -> dict:
    print("Your Conversation has been flagged!, restarting the conversation.")
    return {}


def respond_node(state: State) -> dict:
    print("Assistant: ", state["ai_response"])
    print("\n\n")
    return {}


# Build graph
graph = StateGraph(State)

graph.add_node("check_input_moderation", check_input_moderation)
graph.add_node("intent_check", intent_check_node)
graph.add_node("out_of_scope", out_of_scope_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("rerank", rerank_node)
graph.add_node("chat", chat_node)
graph.add_node("check_ai_moderation", check_ai_moderation)
graph.add_node("flagged", flagged_node)
graph.add_node("respond", respond_node)

graph.set_entry_point("check_input_moderation")

# Routing
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

app = graph.compile()


if __name__ == "__main__":
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SystemInstruction.prompt}
    ]

    while True:
        input_message = input("User: ")
        if input_message in ["exit", "bye", "end"]:
            print("Thank you for your time, hope I helped. Bye!")
            print("Chat Terminated....")
            break

        final_state = app.invoke(
            {
                "messages": messages,
                "input_message": input_message,
            }
        )

        messages = final_state.get("messages", messages)