# From: https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/#prerequisites
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama.chat_models import ChatOllama

# Import utility functions for LangChain/LangSmith setup
from utils import setup_langsmith

# Setup LangSmith tracing (loads .env and displays status)
setup_langsmith()

llm = ChatOllama(
    model="qwen2.5:7b-instruct",
    base_url="http://localhost:11434",
    temperature=0.0,
)

# Start by creating a StateGraph. 
# A StateGraph object defines the structure of our chatbot as a "state machine". 
# We'll add nodes to represent the llm and functions our chatbot can call and edges to specify how the bot should transition between these functions.

# Each node can receive the current State as input and output an update to the state.
#Updates to messages will be appended to the existing list rather than overwriting it, thanks to the prebuilt reducer function.
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break