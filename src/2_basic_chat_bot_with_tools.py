# From: https://langchain-ai.github.io/langgraph/tutorials/get-started/2-add-tools/
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama.chat_models import ChatOllama

# Import utility functions for LangChain/LangSmith and Tavily setup
from utils import setup_langsmith, setup_tavily

# Setup LangSmith tracing (loads .env and displays status)
setup_langsmith()

# Define the State (same as tutorial 1)
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Setup Tavily search tool (following tutorial step 3)
tool = setup_tavily(max_results=2)  # Using max_results=2 as in tutorial
tools = [tool] if tool else []

# Initialize LLM (same as tutorial 1)
llm = ChatOllama(
    model="qwen2.5:7b-instruct",
    base_url="http://localhost:11434",
    temperature=0.0,
)

# Create the graph builder
graph_builder = StateGraph(State)

# Bind tools to LLM (tutorial step 4)
if tools:
    llm_with_tools = llm.bind_tools(tools)
    print(f"🔧 LLM configured with {len(tools)} tool(s)")
    
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    # Add nodes (tutorial steps 4-5)
    graph_builder.add_node("chatbot", chatbot)
    
    # Add ToolNode (tutorial step 9 - using prebuilts)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    # Add conditional edges (tutorial step 6 & 9)
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,  # Using prebuilt tools_condition
    )
    
    # Add edges (tutorial step 6)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    
else:
    print("ℹ️  No tools available - creating basic chatbot")
    
    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}
    
    # Fallback to basic chatbot (same as tutorial 1)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()
print("✅ Graph compiled successfully")


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