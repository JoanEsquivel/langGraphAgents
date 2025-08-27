# From: https://langchain-ai.github.io/langgraph/tutorials/get-started/3-add-memory/
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama.chat_models import ChatOllama

# Import ragas-specific message types for different metrics
try:
    from ragas.messages import HumanMessage as RagasHumanMessage, AIMessage as RagasAIMessage, ToolMessage as RagasToolMessage, ToolCall as RagasToolCall
    from ragas.dataset_schema import MultiTurnSample
    RAGAS_AVAILABLE = True
except ImportError:
    print("⚠️  Ragas not available. Install with: pip install ragas")
    RAGAS_AVAILABLE = False

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
    
    # Print the formatted conversation history after the interaction
    print_formatted_conversation(thread_id)


def get_graph_state(thread_id: str = DEFAULT_THREAD_ID):
    """
    5. Inspect the state (following tutorial step 5)
    Get the current state of the graph for a given thread_id.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    return snapshot


def format_conversation_history(thread_id: str = DEFAULT_THREAD_ID):
    """
    Format and return the conversation history as string representations for display:
    [
        'HumanMessage(content="...")',
        'AIMessage(content="...", tool_calls=[ToolCall(name="...", args={...})])',
        'ToolMessage(content="...")',
        ...
    ]
    """
    snapshot = get_graph_state(thread_id)
    messages = snapshot.values.get("messages", [])
    
    formatted_messages = []
    
    for message in messages:
        # Handle different message types
        if hasattr(message, 'type'):
            if message.type == 'human':
                formatted_messages.append(f'HumanMessage(content="{message.content}")')
            elif message.type == 'ai':
                # Check if the AI message has tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_calls_str = ", ".join([
                        f'ToolCall(name="{tc.get("name", "")}", args={tc.get("args", {})})'
                        for tc in message.tool_calls
                    ])
                    formatted_messages.append(f'AIMessage(content="{message.content}", tool_calls=[{tool_calls_str}])')
                else:
                    formatted_messages.append(f'AIMessage(content="{message.content}")')
            elif message.type == 'tool':
                formatted_messages.append(f'ToolMessage(content="{message.content}")')
        else:
            # Fallback for messages that don't have a type attribute
            if hasattr(message, 'role'):
                if message.role == 'user':
                    formatted_messages.append(f'HumanMessage(content="{message.content}")')
                elif message.role == 'assistant':
                    formatted_messages.append(f'AIMessage(content="{message.content}")')
            else:
                formatted_messages.append(f'Message(content="{getattr(message, "content", str(message))}")')
    
    return formatted_messages


def get_conversation_for_ragas(thread_id: str = DEFAULT_THREAD_ID):
    """
    Return the actual conversation history message objects for agent_topic_adherence metric.
    Can be directly assigned to a variable like: sample_input_4 = get_conversation_for_ragas()
    
    Returns:
        List of actual LangChain message objects (HumanMessage, AIMessage, ToolMessage)
    """
    snapshot = get_graph_state(thread_id)
    messages = snapshot.values.get("messages", [])
    
    # Return the actual message objects, not string representations
    return messages


def get_conversation_for_tool_accuracy(thread_id: str = DEFAULT_THREAD_ID):
    """
    Return conversation in ragas format for ToolCallAccuracy (_agent_tool_accuracy) metric.
    Can be directly assigned to a variable like: sample = get_conversation_for_tool_accuracy()
    
    Returns:
        List of ragas message objects (ragas.messages.HumanMessage, AIMessage, ToolMessage)
    """
    if not RAGAS_AVAILABLE:
        print("❌ Ragas not available. Cannot convert to ragas format.")
        return []
    
    snapshot = get_graph_state(thread_id)
    messages = snapshot.values.get("messages", [])
    
    ragas_messages = []
    
    for message in messages:
        if hasattr(message, 'type'):
            if message.type == 'human':
                ragas_messages.append(RagasHumanMessage(content=message.content))
            elif message.type == 'ai':
                # Check if the AI message has tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_calls = []
                    for tc in message.tool_calls:
                        tool_calls.append(RagasToolCall(
                            name=tc.get("name", ""),
                            args=tc.get("args", {})
                        ))
                    ragas_messages.append(RagasAIMessage(
                        content=message.content,
                        tool_calls=tool_calls
                    ))
                else:
                    ragas_messages.append(RagasAIMessage(content=message.content))
            elif message.type == 'tool':
                ragas_messages.append(RagasToolMessage(content=message.content))
        else:
            # Fallback for messages that don't have a type attribute
            if hasattr(message, 'role'):
                if message.role == 'user':
                    ragas_messages.append(RagasHumanMessage(content=message.content))
                elif message.role == 'assistant':
                    ragas_messages.append(RagasAIMessage(content=message.content))
    
    return ragas_messages


def get_conversation_for_goal_accuracy(thread_id: str = DEFAULT_THREAD_ID):
    """
    Return conversation in MultiTurnSample format for agent_goal_accuracy_with_reference metric.
    Can be directly assigned to a variable like: sample = get_conversation_for_goal_accuracy()
    
    Returns:
        MultiTurnSample object with user_input parameter containing ragas message objects
    """
    if not RAGAS_AVAILABLE:
        print("❌ Ragas not available. Cannot convert to MultiTurnSample format.")
        return None
    
    # Get the ragas-formatted messages
    ragas_messages = get_conversation_for_tool_accuracy(thread_id)
    
    # Wrap in MultiTurnSample with user_input parameter
    return MultiTurnSample(user_input=ragas_messages)


def print_formatted_conversation(thread_id: str = DEFAULT_THREAD_ID):
    """
    Print the conversation history in the requested format.
    """
    formatted_messages = format_conversation_history(thread_id)
    
    if formatted_messages:
        print("\n" + "="*80)
        print("📜 FORMATTED CONVERSATION HISTORY:")
        print("="*80)
        print("[")
        for msg in formatted_messages:
            print(f"    {msg},")
        print("]")
        print("="*80)


# Demo the memory functionality following the tutorial examples
if __name__ == "__main__":
    print("\n🧠 LangGraph Chatbot with Memory - Tutorial Example")
    print("=" * 50)
    print("Commands:")
    print("  'quit', 'exit', 'q' - Exit the program")
    print("  'thread2' - Switch to conversation thread 2")
    print("  'thread1' - Switch back to conversation thread 1")
    print("  'state' - Show current conversation state")
    print("  'history' - Show formatted conversation history")
    print("  'ragas' - Get LangChain objects for agent_topic_adherence")
    print("  'tool_accuracy' - Get ragas objects for _agent_tool_accuracy")
    print("  'goal_accuracy' - Get MultiTurnSample for agent_goal_accuracy_with_reference")
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
            elif user_input.lower() == "history":
                print_formatted_conversation(current_thread)
                continue
            elif user_input.lower() == "ragas":
                print(f"🔬 LangChain objects for agent_topic_adherence (thread {current_thread}):")
                messages = get_conversation_for_ragas(current_thread)
                print(f"sample_input_{current_thread} = {messages}")
                print(f"\n📋 Use for agent_topic_adherence metric")
                continue
            elif user_input.lower() == "tool_accuracy":
                print(f"🔧 Ragas objects for _agent_tool_accuracy (thread {current_thread}):")
                messages = get_conversation_for_tool_accuracy(current_thread)
                print(f"sample_tool_accuracy_{current_thread} = {messages}")
                print(f"\n📋 Use for ToolCallAccuracy (_agent_tool_accuracy) metric")
                continue
            elif user_input.lower() == "goal_accuracy":
                print(f"🎯 MultiTurnSample for agent_goal_accuracy_with_reference (thread {current_thread}):")
                sample = get_conversation_for_goal_accuracy(current_thread)
                print(f"sample_goal_accuracy_{current_thread} = {sample}")
                print(f"\n📋 Use for agent_goal_accuracy_with_reference metric")
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