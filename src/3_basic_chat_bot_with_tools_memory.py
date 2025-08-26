# From: https://langchain-ai.github.io/langgraph/tutorials/get-started/3-add-memory/
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama.chat_models import ChatOllama

# Import utility functions for LangChain/LangSmith and Tavil y setup
from utils import setup_langsmith, setup_tavily

# Setup LangSmith tracing (loads .env and displays status)
setup_langsmith()

# 1. Create a MemorySaver checkpointer (following tutorial step 1)
memory = InMemorySaver()

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

# 2. Compile the graph with checkpointer (following tutorial step 2)
graph = graph_builder.compile(checkpointer=memory)
print("✅ Graph compiled successfully with memory checkpointing")

# Default thread ID for conversations
DEFAULT_THREAD_ID = "1"

def stream_graph_updates(user_input: str, thread_id: str = DEFAULT_THREAD_ID):
    """
    Stream graph updates for a user input following the tutorial pattern.
    Uses thread_id for memory management across conversations.
    """
    # 3. Interact with your chatbot (following tutorial step 3)
    config = {"configurable": {"thread_id": thread_id}}
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        # Follow the tutorial pattern: event["messages"][-1].pretty_print()
        if isinstance(event, dict) and "messages" in event:
            event["messages"][-1].pretty_print()


def get_graph_state(thread_id: str = DEFAULT_THREAD_ID):
    """
    5. Inspect the state (following tutorial step 5)
    Get the current state of the graph for a given thread_id.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    return snapshot


# Demo the memory functionality following the tutorial examples
if __name__ == "__main__":
    print("\n🧠 LangGraph Chatbot with Memory - Tutorial Example")
    print("=" * 50)
    print("Commands:")
    print("  'quit', 'exit', 'q' - Exit the program")
    print("  'thread2' - Switch to conversation thread 2")
    print("  'thread1' - Switch back to conversation thread 1")
    print("  'state' - Show current conversation state")
    print("=" * 50)
    
    current_thread = DEFAULT_THREAD_ID
    
    print(f"\n💬 Interactive mode (Thread {current_thread})")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "thread2":
                current_thread = "2"
                print(f"📋 Switched to conversation thread {current_thread}")
                continue
            elif user_input.lower() == "thread1":
                current_thread = "1"
                print(f"📋 Switched to conversation thread {current_thread}")
                continue
            elif user_input.lower() == "state":
                print(f"📊 Current state for thread {current_thread}:")
                snapshot = get_graph_state(current_thread)
                print(f"Messages count: {len(snapshot.values.get('messages', []))}")
                print(f"Thread ID: {snapshot.config['configurable']['thread_id']}")
                continue
            
            stream_graph_updates(user_input, current_thread)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input, current_thread)
            break