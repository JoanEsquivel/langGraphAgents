# LangGraph Agents Learning Project

A comprehensive tutorial project for building AI agents using LangChain and LangGraph, following the official tutorials with practical implementations.

## 🎯 Overview

This project demonstrates the progressive evolution from basic chatbots to advanced AI agents with tool integration and persistent memory:

- **LangChain** provides the foundational building blocks (LLMs, prompts, retrievers, agents) for AI applications
- **LangGraph** orchestrates these components as a graph, enabling stateful, multi-step, and controllable agent workflows
- **Progression**: Each tutorial builds upon the previous one, adding complexity and capabilities

## 📈 Evolution Diagram

```mermaid
graph LR
    subgraph "Evolution of LangGraph Chatbots"
        subgraph "Tutorial 1: Basic Chatbot"
            A1["User Input"] --> B1["Chatbot Node"]
            B1 --> C1["LLM"]
            C1 --> D1["Response"]
        end
        
        subgraph "Tutorial 2: + Tools"
            A2["User Input"] --> B2["Chatbot Node"]
            B2 --> C2{{"tools_condition"}}
            C2 -->|Tool needed| D2["Tools Node"]
            C2 -->|No tools| E2["END"]
            D2 --> F2["Web Search"]
            F2 --> B2
        end
        
        subgraph "Tutorial 3: + Memory"
            A3["User Input<br/>Thread ID"] --> B3["Chatbot Node"]
            B3 --> C3{{"Memory Check<br/>+ tools_condition"}}
            C3 -->|Load State| D3["Previous Context"]
            C3 -->|Tool needed| E3["Tools Node"]
            C3 -->|Response ready| F3["Save State"]
            D3 --> B3
            E3 --> G3["Web Search"]
            G3 --> B3
            F3 --> H3["Response"]
        end
    end
    
    style A1 fill:#ffcdd2
    style B1 fill:#f8bbd9
    style C1 fill:#e1bee7
    style D1 fill:#d1c4e9
    
    style A2 fill:#fff3e0
    style B2 fill:#ffe0b2
    style C2 fill:#ffcc02
    style D2 fill:#ffb74d
    style E2 fill:#ff8a65
    style F2 fill:#ff7043
    
    style A3 fill:#e8f5e8
    style B3 fill:#c8e6c9
    style C3 fill:#a5d6a7
    style D3 fill:#81c784
    style E3 fill:#66bb6a
    style F3 fill:#4caf50
    style G3 fill:#43a047
    style H3 fill:#388e3c
```

## 📚 Tutorials Implemented

| Tutorial | Script | Description | Key Features |
|----------|--------|-------------|--------------|
| [Tutorial 1](https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/) | `src/1_basic_chat_bot.py` | Basic stateful chatbot | 🔄 Message history, 📝 State management |
| [Tutorial 2](https://langchain-ai.github.io/langgraph/tutorials/get-started/2-add-tools/) | `src/2_basic_chat_bot_with_tools.py` | Advanced chatbot with web search | 🔍 Tool integration, 🤖 Intelligent routing |
| [Tutorial 3](https://langchain-ai.github.io/langgraph/tutorials/get-started/3-add-memory/) | `src/3_basic_chat_bot_with_tools_memory.py` | **Memory-enabled chatbot** | 🧠 **Persistent memory**, 🔀 **Multi-conversation** |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) running locally with `qwen2.5:7b-instruct` model
- Optional: LangSmith API key for tracing
- Optional: Tavily API key for web search (Tutorials 2-3)

### Installation

1. **Create and activate virtual environment:**
   ```bash
   pyenv install 3.11
   pyenv shell 3.11
   python -m venv venv_foundational_agents
   source venv_foundational_agents/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment (optional):**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

### Quick Test

```bash
# Test basic chatbot
python src/1_basic_chat_bot.py

# Test advanced chatbot with tools
python src/2_basic_chat_bot_with_tools.py

# Test memory-enabled chatbot (NEW!)
python src/3_basic_chat_bot_with_tools_memory.py
```

## 📁 Project Structure

```
langGraphAgents/
├── src/
│   ├── 1_basic_chat_bot.py                    # Tutorial 1: Basic stateful chatbot
│   ├── 2_basic_chat_bot_with_tools.py         # Tutorial 2: Chatbot with web search
│   ├── 3_basic_chat_bot_with_tools_memory.py  # Tutorial 3: Memory-enabled chatbot
│   └── utils/                                 # Utility modules
│       ├── __init__.py
│       ├── langchain_setup.py                 # LangChain/LangSmith configuration
│       └── tavily_setup.py                    # Tavily search tool setup
├── env.example                                # Environment variables template
├── requirements.txt                           # Python dependencies
└── README.md                                  # This file
```

## 🤖 Tutorial 1: Basic Chatbot

### Overview

The `src/1_basic_chat_bot.py` implements a foundational stateful chatbot using LangGraph that maintains conversation history and connects to a local Ollama model.

### Architecture Diagram

```mermaid
graph LR
    A[User Input] --> B[Chatbot Node]
    B --> C[Ollama LLM]
    C --> D[Response]
    D --> E[Update State]
    E --> F[Display to User]
    
    subgraph "State Management"
        G[Message History]
        H[add_messages reducer]
    end
    
    B --> G
    G --> H
    H --> B
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#e8f5e8
    style E fill:#fff9c4
    style F fill:#fce4ec
```

### Key Features

- **🔄 Stateful Conversations**: Maintains message history across interactions
- **📝 Message Accumulation**: Uses `add_messages` reducer to append (not overwrite) messages
- **🚀 Simple Architecture**: Linear flow from input to response
- **📊 LangSmith Integration**: Optional tracing for monitoring conversations

### Implementation Details

#### State Management
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```
- **State**: TypedDict structure maintaining conversation history
- **add_messages**: Built-in reducer ensuring message accumulation (not replacement)

#### Core Components
```python
# LLM Configuration
llm = ChatOllama(
    model="qwen2.5:7b-instruct",
    base_url="http://localhost:11434",
    temperature=0.0,
)

# Graph Structure
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()
```

#### Execution Flow
1. **User Input** → **Chatbot Node** → **Ollama LLM** → **Response Generation**
2. **State Update** → **Message History** → **Display Result**

### Usage Example

```bash
$ python src/1_basic_chat_bot.py
✅ Loaded environment variables from /path/to/.env
🔍 LangSmith tracing enabled for project: langgraphagents
User: Hello, how are you?
Assistant: Hello! I'm doing well, thank you for asking. I'm here and ready to help...

User: What can you help me with?
Assistant: I can assist you with a wide variety of tasks and questions...

User: quit
Goodbye!
```

## 🔧 Tutorial 2: Advanced Chatbot with Tools

### Overview

The `src/2_basic_chat_bot_with_tools.py` implements an advanced chatbot that can use external tools, specifically web search via Tavily, following the official LangGraph tutorial pattern.

### Architecture Diagram

```mermaid
graph TD
    A[User Input] --> B[Chatbot Node]
    B --> C{tools_condition}
    C -->|Has tool_calls| D[Tools Node]
    C -->|No tool_calls| E[END]
    D --> F[Execute Tavily Search]
    F --> G[Format Results]
    G --> B
    
    subgraph "LLM Processing"
        H[Analyze Query]
        I[Decide Tool Usage]
        J[Generate Response]
    end
    
    subgraph "Tool Execution"
        K[Tavily API]
        L[Web Search]
        M[Result Processing]
    end
    
    B --> H
    H --> I
    I --> J
    D --> K
    K --> L
    L --> M
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#e8f5e8
    style E fill:#ffebee
    style F fill:#fff9c4
```

### Key Features

- **🔍 Web Search Integration**: Uses Tavily API for real-time web searches
- **🤖 Intelligent Routing**: Automatically decides when to use tools
- **🔧 Prebuilt Components**: Uses LangGraph's optimized ToolNode and tools_condition
- **📊 Full Tracing**: LangSmith integration for monitoring tool usage
- **⚡ Graceful Fallback**: Works as basic chatbot if tools are unavailable

### Usage Example

```bash
$ python src/2_basic_chat_bot_with_tools.py
✅ Loaded environment variables from /path/to/.env
🔍 LangSmith tracing enabled for project: langgraphagents
🔍 Tavily API key configured: tvly-dev...
✅ Tavily search tool configured (max_results=2)
🔧 LLM configured with 1 tool(s)
✅ Graph compiled successfully

User: What are the latest updates in LangGraph 2025?
Assistant: [Performs web search via Tavily]
Based on the search results, here are some of the latest updates in LangGraph 2025:

### Node Caching (♻️)
LangChain introduced node/task level caching, which allows you to cache...

User: What is 2+2?
Assistant: 2+2 equals 4. This is a basic arithmetic operation.

User: quit
Goodbye!
```

## 🧠 Tutorial 3: Memory-Enabled Chatbot (NEW!)

### Overview

The `src/3_basic_chat_bot_with_tools_memory.py` implements a **memory-enabled chatbot** that can maintain persistent conversation state across multiple sessions and different conversation threads. This follows the [LangGraph Memory Tutorial](https://langchain-ai.github.io/langgraph/tutorials/get-started/3-add-memory/).

### Architecture Diagram

```mermaid
graph TD
    A["User Input<br/>💬"] --> B["Chatbot Node<br/>🤖"]
    B --> C{{"Memory Checkpoint<br/>🧠<br/>Thread: {thread_id}"}}
    C --> D["Ollama LLM<br/>🤖"]
    D --> E["Response Generation<br/>💭"]
    E --> F["Update State<br/>📝"]
    F --> G["Save Checkpoint<br/>💾"]
    G --> H["Display to User<br/>📺"]
    
    subgraph "Memory System"
        I["InMemorySaver<br/>💾"]
        J["Thread 1 State<br/>📊"]
        K["Thread 2 State<br/>📊"]
        L["Thread N State<br/>📊"]
    end
    
    subgraph "State Management"
        M["Message History<br/>📚"]
        N["add_messages reducer<br/>➕"]
    end
    
    C --> I
    I --> J
    I --> K
    I --> L
    B --> M
    M --> N
    N --> B
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#e8f5e8
    style E fill:#fff9c4
    style F fill:#fce4ec
    style G fill:#e0f2f1
    style H fill:#f3e5f5
```

### Key Features

- **🧠 Persistent Memory**: Maintains conversation context across sessions using LangGraph checkpointing
- **🔀 Multi-Conversation Support**: Manages multiple independent conversation threads
- **🔍 Tool Integration**: Combines memory with web search capabilities
- **📊 State Inspection**: Built-in functionality to inspect conversation state
- **⚡ Thread Management**: Easy switching between different conversation contexts
- **💾 Automatic Checkpointing**: Saves state after each interaction automatically

### Memory Demonstration

```mermaid
graph TD
    A["Thread 1: User says<br/>'Hi! My name is Will'"] --> B["Chatbot processes<br/>and responds"]
    B --> C["State saved with<br/>thread_id: '1'"]
    C --> D["Thread 1: User asks<br/>'Remember my name?'"]
    D --> E["Chatbot loads previous<br/>context from thread 1"]
    E --> F["Response: 'Yes, Will!'<br/>✅ Memory works"]
    
    G["Thread 2: User asks<br/>'Remember my name?'"] --> H["Chatbot checks<br/>thread_id: '2'"]
    H --> I["No previous context<br/>found for thread 2"]
    I --> J["Response: 'I don't know<br/>your name' ❌ Clean slate"]
    
    K["Thread 1: User asks<br/>'What's my name?'"] --> L["Chatbot loads context<br/>from thread 1 again"]
    L --> M["Response: 'Your name<br/>is Will' ✅ Persistent memory"]
    
    subgraph "Memory Storage"
        N["InMemorySaver"]
        O["Thread 1 Messages<br/>• Hi! My name is Will<br/>• Hello Will!<br/>• Remember my name?<br/>• Yes, Will!"]
        P["Thread 2 Messages<br/>• Remember my name?<br/>• I don't know your name"]
    end
    
    C --> N
    E --> O
    H --> N
    I --> P
    L --> O
    
    style A fill:#e3f2fd
    style D fill:#e3f2fd
    style K fill:#e3f2fd
    style G fill:#fff3e0
    style F fill:#e8f5e8
    style J fill:#ffebee
    style M fill:#e8f5e8
    style N fill:#f3e5f5
    style O fill:#e1f5fe
    style P fill:#fff9c4
```

### Implementation Details

#### 1. Memory Checkpointer
```python
from langgraph.checkpoint.memory import InMemorySaver

# 1. Create a MemorySaver checkpointer (following tutorial step 1)
memory = InMemorySaver()
```

#### 2. Graph Compilation with Memory
```python
# 2. Compile the graph with checkpointer (following tutorial step 2)
graph = graph_builder.compile(checkpointer=memory)
```

#### 3. Thread-based Configuration
```python
def stream_graph_updates(user_input: str, thread_id: str = DEFAULT_THREAD_ID):
    # 3. Interact with your chatbot (following tutorial step 3)
    config = {"configurable": {"thread_id": thread_id}}
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,  # The config is the **second positional argument**!
        stream_mode="values",
    )
```

#### 4. State Inspection
```python
def get_graph_state(thread_id: str = DEFAULT_THREAD_ID):
    # 5. Inspect the state (following tutorial step 5)
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    return snapshot
```

### Memory Benefits

- ✅ **Persistent Context**: Conversations continue where they left off
- ✅ **Multiple Sessions**: Different users or topics can have separate threads
- ✅ **Automatic Management**: No manual memory handling required
- ✅ **Production Ready**: Can easily switch to SqliteSaver or PostgresSaver
- ✅ **State Recovery**: Full conversation state can be inspected and restored
- ✅ **Error Recovery**: Checkpointing enables robust error handling

### Advanced Features

#### Interactive Commands
- `thread1` / `thread2` - Switch between conversation threads
- `state` - Inspect current conversation state
- `quit` - Exit the program

#### Memory Types
- **InMemorySaver**: Tutorial/development use (data lost on restart)
- **SqliteSaver**: Production use with SQLite database
- **PostgresSaver**: Production use with PostgreSQL database

### Usage Example

```bash
$ python src/3_basic_chat_bot_with_tools_memory.py
✅ Loaded environment variables from /path/to/.env
🔍 LangSmith tracing enabled for project: langgraphagents
🔍 Tavily API key configured: tvly-dev...
✅ Tavily search tool configured (max_results=2)
✅ Tavily tool ready for web search
🔧 LLM configured with 1 tool(s)
✅ Graph compiled successfully with memory checkpointing

🧠 LangGraph Chatbot with Memory - Tutorial Example
==================================================
Commands:
  'quit', 'exit', 'q' - Exit the program
  'thread2' - Switch to conversation thread 2
  'thread1' - Switch back to conversation thread 1
  'state' - Show current conversation state
==================================================

🔬 Demo: Testing memory functionality (Thread 1)
User: Hi there! My name is Will.
================================ Human Message =================================

Hi there! My name is Will.
================================== Ai Message ==================================

Hello, Will! Nice to meet you. How can I assist you today?

User: Remember my name?
================================ Human Message =================================

Remember my name?
================================== Ai Message ==================================

Of course! You mentioned your name is Will. Is there anything specific you'd like to discuss?

🔬 Demo: Testing different thread (Thread 2)
User: Remember my name?
================================ Human Message =================================

Remember my name?
================================== Ai Message ==================================

I apologize, but I don't have any previous context or memory of your name. Could you please tell me your name?

💬 Interactive mode (Thread 1)
Try asking about your name again to see memory working!
User: what is my name
================================ Human Message =================================

what is my name
================================== Ai Message ==================================

Your name is Will. How can I assist you further with that, Will?

User: thread2
📋 Switched to conversation thread 2
User: my name is John
================================ Human Message =================================

my name is John
================================== Ai Message ==================================

Hello John! It's nice to meet you. How can I help you today?

User: thread1
📋 Switched to conversation thread 1
User: remember my name?
================================ Human Message =================================

remember my name?
================================== Ai Message ==================================

Yes, your name is Will. Is there anything else you'd like to discuss?

User: quit
Goodbye!
```

## 🔧 Environment Configuration

### LangSmith Tracing (Optional)

For monitoring and debugging conversations:

1. **Copy the environment template:**
   ```bash
   cp env.example .env
   ```

2. **Edit the `.env` file with your API credentials:**
   ```bash
   # LangSmith Configuration (Optional - for tracing)
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=your_langsmith_api_key_here
   LANGSMITH_PROJECT=langraph-chatbot
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   
   # Tavily Configuration (Optional - for web search in tutorials 2-3)
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

3. **Get your API keys:**
   - **LangSmith**: [smith.langchain.com](https://smith.langchain.com/) (for tracing)
   - **Tavily**: [tavily.com](https://tavily.com/) (for web search in tutorials 2-3)

### Environment Features

The setup automatically:
- ✅ Loads environment variables from `.env` file (if it exists)
- 🔧 Supports both `LANGSMITH_` and `LANGCHAIN_` variable prefixes
- 🔄 Maps LangSmith variables to LangChain tracing internally
- 🔍 Displays tracing status on startup
- 📊 Sends traces to LangSmith when configured
- ⚠️ Works normally without LangSmith (tracing disabled)

### Tavily Configuration

To enable web search capabilities in tutorials 2-3:

1. **Get Tavily API Key**: Visit [tavily.com](https://tavily.com/) to get your API key
2. **Add to `.env`**: Include `TAVILY_API_KEY=your_api_key_here`
3. **Tool Status**: The script will show:
   - ✅ `Tavily search tool configured` if API key is valid
   - ⚠️ `TAVILY_API_KEY not found` if not configured

**Without Tavily**: The chatbot will still work but without web search capabilities.

## 🎓 Learning Path

### Recommended Order

1. **🚀 Start with Tutorial 1**: Understand basic LangGraph concepts
   - Learn about `StateGraph`, message management, and simple flows
   - Get comfortable with the basic chat interface

2. **🔧 Progress to Tutorial 2**: Add tool integration
   - Understand conditional edges and tool routing
   - Learn about `ToolNode` and `tools_condition` prebuilt components
   - Experience web search integration

3. **🧠 Master Tutorial 3**: Implement persistent memory
   - Understand checkpointing and state persistence
   - Learn about thread management and multi-conversation support
   - Combine memory with tool usage for powerful agents

### Key Concepts Learned

- **StateGraph**: Core LangGraph orchestration
- **Message Management**: Using `add_messages` reducer
- **Tool Integration**: Adding external capabilities
- **Conditional Routing**: Intelligent decision making
- **Memory/Checkpointing**: Persistent conversation state
- **Thread Management**: Multiple conversation contexts
- **Production Considerations**: Database backends, error handling

## 📖 Learning Resources

- [LangGraph Concepts](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)
- [Tutorial 1: Build a Basic Chatbot](https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/)
- [Tutorial 2: Add Tools](https://langchain-ai.github.io/langgraph/tutorials/get-started/2-add-tools/)
- [Tutorial 3: Add Memory](https://langchain-ai.github.io/langgraph/tutorials/get-started/3-add-memory/)
- [Tutorial 4: Human-in-the-Loop](https://langchain-ai.github.io/langgraph/tutorials/get-started/4-human-in-the-loop/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Ollama Setup Guide](https://ollama.ai/)

## 🔮 Next Steps

After completing these tutorials, consider exploring:

- **Tutorial 4**: Human-in-the-loop workflows
- **Tutorial 5**: Custom state management
- **Tutorial 6**: Time travel debugging
- **Production Deployment**: Using PostgreSQL/SQLite for persistence
- **Advanced Routing**: Custom conditional logic
- **Multi-Agent Systems**: Agent collaboration patterns

## 🤝 Contributing

This project follows the official LangGraph tutorials. Contributions that improve clarity, add examples, or fix issues are welcome.

## 📄 License

This project is for educational purposes, following the MIT License.