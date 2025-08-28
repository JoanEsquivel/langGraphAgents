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

# Initialize LLM with QA automation system prompt
system_prompt = """You are a QA automation expert and consultant. Your role is to provide focused, professional advice specifically about:

- Test automation frameworks and tools
- QA testing strategies and best practices  
- Software testing methodologies
- Performance, API, functional, mobile testing
- CI/CD testing integration
- Test data management and validation

IMPORTANT CONSTRAINTS:
- Stay strictly within QA automation and software testing topics
- If asked about unrelated topics (weather, cooking, general knowledge), politely redirect to QA automation
- Provide focused, actionable recommendations backed by current industry knowledge
- When recommending tools, provide specific implementation guidance

Example redirect: "I specialize in QA automation. Let me help you with testing strategies instead. What specific testing challenges are you facing?"
"""

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
        # Add system prompt as first message if not present
        messages = state["messages"]
        if not messages or not (hasattr(messages[0], 'type') and messages[0].type == "system"):
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=system_prompt)] + messages
        return {"messages": [llm_with_tools.invoke(messages)]}
    
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


def _convert_to_ragas_messages(thread_id: str = DEFAULT_THREAD_ID):
    """
    Internal helper function to convert conversation to RAGAS format.
    
    Filters out system messages and cleans content for RAGAS compatibility.
    
    Returns:
        List of RAGAS message objects (ragas.messages.HumanMessage, AIMessage, ToolMessage)
    """
    snapshot = get_graph_state(thread_id)
    messages = snapshot.values.get("messages", [])
    
    ragas_messages = []
    
    for message in messages:
        if hasattr(message, 'type'):
            # Skip system messages as they can confuse RAGAS evaluation
            if message.type == 'system':
                continue
                
            if message.type == 'human':
                # Clean and normalize human message content
                content = message.content.strip()
                if content:  # Only add non-empty messages
                    ragas_messages.append(RagasHumanMessage(content=content))
                    
            elif message.type == 'ai':
                # Clean and normalize AI message content
                content = message.content.strip() if message.content else ""
                
                # Check if the AI message has tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_calls = []
                    for tc in message.tool_calls:
                        # Ensure tool call data is properly formatted
                        tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, 'name', '')
                        tool_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                        
                        # Only add valid tool calls
                        if tool_name:
                            tool_calls.append(RagasToolCall(
                                name=str(tool_name),
                                args=dict(tool_args) if tool_args else {}
                            ))
                    
                    # Always add AI messages with tool calls, even if content is empty
                    # This is required by RAGAS - ToolMessage must be preceded by AIMessage
                    ragas_messages.append(RagasAIMessage(
                        content=content if content else "",
                        tool_calls=tool_calls
                    ))
                else:
                    # Only add AI messages with content
                    if content:
                        ragas_messages.append(RagasAIMessage(content=content))
                        
            elif message.type == 'tool':
                # Clean tool message content and ensure it's properly formatted
                content = message.content.strip() if message.content else ""
                if content:
                    # Ensure tool message content is a clean string, not complex JSON
                    try:
                        import json
                        # If content is JSON, extract the key information for RAGAS
                        if content.startswith('{') and content.endswith('}'):
                            parsed = json.loads(content)
                            # Extract main content, avoiding complex nested structures
                            if 'query' in parsed and 'results' in parsed:
                                # Summarize search results for RAGAS
                                query = parsed.get('query', '')
                                results_count = len(parsed.get('results', []))
                                content = f"Search query: {query}, Found {results_count} results"
                            elif isinstance(parsed, dict):
                                # Extract first meaningful value
                                content = str(list(parsed.values())[0]) if parsed else content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Keep original content if JSON parsing fails
                        pass
                    
                    ragas_messages.append(RagasToolMessage(content=content))
        else:
            # Fallback for messages that don't have a type attribute
            if hasattr(message, 'role'):
                # Skip system role messages
                if getattr(message, 'role', '') == 'system':
                    continue
                    
                content = getattr(message, 'content', '').strip()
                if content:  # Only add non-empty messages
                    if message.role == 'user':
                        ragas_messages.append(RagasHumanMessage(content=content))
                    elif message.role == 'assistant':
                        ragas_messages.append(RagasAIMessage(content=content))
    
    return ragas_messages


def getMultiTurnSampleConversation(thread_id: str = DEFAULT_THREAD_ID):
    """
    UNIFIED METHOD: Return conversation in MultiTurnSample format for all RAGAS metrics.
    
    This method replaces the three previous methods:
    - get_conversation_for_ragas()
    - get_conversation_for_tool_accuracy() 
    - get_conversation_for_goal_accuracy()
    
    Usage Examples:
    - Topic Adherence: sample = getMultiTurnSampleConversation(thread_id)
                      sample.reference_topics = ["weather", "testing"]
    - Tool Accuracy:   sample = getMultiTurnSampleConversation(thread_id)
                      sample.reference_tool_calls = extracted_tool_calls
    - Goal Accuracy:   sample = getMultiTurnSampleConversation(thread_id)
                      sample.reference = "goal description"
    
    Args:
        thread_id (str): Conversation thread identifier
    
    Returns:
        MultiTurnSample object with user_input containing RAGAS message objects
    """
    if not RAGAS_AVAILABLE:
        print("❌ Ragas not available. Cannot convert to MultiTurnSample format.")
        return None
    
    # Get the ragas-formatted messages with cleaned content
    ragas_messages = _convert_to_ragas_messages(thread_id)
    
    # Validate that we have meaningful conversation data
    if not ragas_messages:
        print("⚠️  No valid messages found for RAGAS evaluation")
        return None
    
    # Ensure proper message sequence for RAGAS 
    validated_messages = []
    for msg in ragas_messages:
        try:
            # Ensure message content is properly formatted string
            if hasattr(msg, 'content'):
                # Clean any problematic characters that might confuse RAGAS
                content = str(msg.content).strip()
                
                # Create new message with cleaned content
                if isinstance(msg, RagasHumanMessage):
                    if content:  # Only add non-empty human messages
                        validated_messages.append(RagasHumanMessage(content=content))
                elif isinstance(msg, RagasAIMessage):
                    # Preserve tool calls if present - ALWAYS add AI messages with tool calls
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        validated_messages.append(RagasAIMessage(
                            content=content if content else "", 
                            tool_calls=msg.tool_calls
                        ))
                    elif content:  # Only add AI messages without tool calls if they have content
                        validated_messages.append(RagasAIMessage(content=content))
                elif isinstance(msg, RagasToolMessage):
                    if content:  # Only add non-empty tool messages
                        validated_messages.append(RagasToolMessage(content=content))
        except Exception as e:
            print(f"⚠️  Skipping invalid message: {e}")
            continue
    
    if not validated_messages:
        print("⚠️  No valid messages after validation for RAGAS evaluation")
        return None
    
    print(f"✅ Created MultiTurnSample with {len(validated_messages)} validated messages")
    
    # Wrap in MultiTurnSample with user_input parameter
    return MultiTurnSample(user_input=validated_messages)


# Backward compatibility functions (deprecated - use getMultiTurnSampleConversation instead)
def get_conversation_for_ragas(thread_id: str = DEFAULT_THREAD_ID):
    """DEPRECATED: Use getMultiTurnSampleConversation() instead."""
    print("⚠️  get_conversation_for_ragas() is deprecated. Use getMultiTurnSampleConversation() instead.")
    if not RAGAS_AVAILABLE:
        return []
    return _convert_to_ragas_messages(thread_id)

def get_conversation_for_tool_accuracy(thread_id: str = DEFAULT_THREAD_ID):
    """DEPRECATED: Use getMultiTurnSampleConversation() instead."""
    print("⚠️  get_conversation_for_tool_accuracy() is deprecated. Use getMultiTurnSampleConversation() instead.")
    if not RAGAS_AVAILABLE:
        return []
    return _convert_to_ragas_messages(thread_id)

def get_conversation_for_goal_accuracy(thread_id: str = DEFAULT_THREAD_ID):
    """DEPRECATED: Use getMultiTurnSampleConversation() instead."""
    print("⚠️  get_conversation_for_goal_accuracy() is deprecated. Use getMultiTurnSampleConversation() instead.")
    return getMultiTurnSampleConversation(thread_id)


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
    print("  'unified' - Get unified MultiTurnSample for ALL RAGAS metrics")
    print("  'ragas' - [DEPRECATED] Get LangChain objects for agent_topic_adherence")
    print("  'tool_accuracy' - [DEPRECATED] Get ragas objects for _agent_tool_accuracy")
    print("  'goal_accuracy' - [DEPRECATED] Get MultiTurnSample for agent_goal_accuracy_with_reference")
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
            elif user_input.lower() == "unified":
                print(f"🔄 UNIFIED MultiTurnSample for ALL RAGAS metrics (thread {current_thread}):")
                sample = getMultiTurnSampleConversation(current_thread)
                print(f"sample_{current_thread} = {sample}")
                print(f"\n📋 Usage:")
                print(f"   • Topic Adherence: sample.reference_topics = ['weather', 'testing']")
                print(f"   • Tool Accuracy:   sample.reference_tool_calls = extracted_tool_calls")
                print(f"   • Goal Accuracy:   sample.reference = 'goal description'")
                print(f"\n✅ Works with ALL RAGAS metrics!")
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