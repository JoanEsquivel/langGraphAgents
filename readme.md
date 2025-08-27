# LangGraph Agent Testing with RAGAS Evaluation Framework

> **📅 Last Updated: August 27, 2025** | **🚀 Production Ready** | **🧪 100% Real Data Testing**

A comprehensive LangGraph agent implementation with industry-standard RAGAS evaluation for QA professionals. This project demonstrates building AI agents using LangChain and LangGraph, followed by rigorous testing using the RAGAS (Retrieval Augmented Generation Assessment) framework.

**✨ Latest Features**: Production agent with multi-format RAGAS integration, enhanced test logging, and zero-simulation architecture for authentic AI evaluation.

## 🎯 Project Overview

This project demonstrates the complete lifecycle from AI agent development to production-ready testing:

- **🎓 Progressive Learning**: Evolution from basic chatbots to production-ready agents with memory
- **🚀 Production Implementation**: `4-final-agent-formated-response.py` with integrated RAGAS formatting  
- **🧪 RAGAS Integration**: Industry-standard evaluation with three specialized conversation formatters
- **📊 Enhanced Testing**: Comprehensive logging, step-by-step process visibility, and metric explanations
- **⚡ Zero Simulation**: 100% authentic AI responses with real tool execution and web searches
- **🎯 QA Best Practices**: Professional thresholds, detailed interpretations, and actionable insights

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
        
        subgraph "Tutorial 3: + Memory + RAGAS Testing"
            A3["User Input<br/>Thread ID"] --> B3["Chatbot Node"]
            B3 --> C3{{"Memory Check<br/>+ tools_condition"}}
            C3 -->|Load State| D3["Previous Context"]
            C3 -->|Tool needed| E3["Tools Node"]
            C3 -->|Response ready| F3["Save State"]
            D3 --> B3
            E3 --> G3["Tavily Search"]
            G3 --> B3
            F3 --> H3["Response"]
            H3 --> I3["RAGAS Evaluation"]
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

## 📚 Tutorial Implementation

| Tutorial | Script | Description | Key Features |
|----------|--------|-------------|--------------|
| [Tutorial 1](https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/) | `src/1_basic_chat_bot.py` | Basic stateful chatbot | 🔄 Message history, 📝 State management |
| [Tutorial 2](https://langchain-ai.github.io/langgraph/tutorials/get-started/2-add-tools/) | `src/2_basic_chat_bot_with_tools.py` | Advanced chatbot with web search | 🔍 Tool integration, 🤖 Intelligent routing |
| [Tutorial 3](https://langchain-ai.github.io/langgraph/tutorials/get-started/3-add-memory/) | `src/3_basic_chat_bot_with_tools_memory.py` | **Memory-enabled chatbot** | 🧠 **Persistent memory**, 🔀 **Multi-conversation** |
| **RAGAS Testing** | `tests/test_real_agent.py` | **Production Testing Suite** | 🧪 **RAGAS evaluation**, ⚡ **Real agent testing** |

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

# Test memory-enabled chatbot
python src/3_basic_chat_bot_with_tools_memory\ copy.py

# 🆕 Test production agent with RAGAS integration (RECOMMENDED)
python src/4-final-agent-formated-response.py

# 🆕 Run comprehensive RAGAS evaluation
pytest tests/test_real_agent.py -v -s
```

## 📁 Project Structure

```
langGraphAgents/
├── src/
│   ├── 1_basic_chat_bot.py                           # Tutorial 1: Basic stateful chatbot
│   ├── 2_basic_chat_bot_with_tools.py                # Tutorial 2: Chatbot with web search
│   ├── 3_basic_chat_bot_with_tools_memory copy.py    # Tutorial 3: Memory-enabled chatbot
│   ├── 4-final-agent-formated-response.py            # 🆕 Production agent with RAGAS integration
│   └── utils/                                        # Utility modules
│       ├── __init__.py
│       ├── langchain_setup.py                        # LangChain/LangSmith configuration
│       └── tavily_setup.py                           # Tavily search tool setup
├── tests/
│   ├── conftest.py                                   # RAGAS evaluation configuration
│   ├── test_real_agent.py                            # 🆕 Comprehensive RAGAS test suite
│   └── helpers/
│       └── utils.py                                  # Test utility functions
├── env.example                                       # Environment variables template
├── requirements.txt                                  # Python dependencies
└── README.md                                         # This file
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

## 🚀 Production Agent: RAGAS-Integrated Implementation

### Overview

The `src/4-final-agent-formated-response.py` represents the **production-ready agent** that combines all tutorial concepts with comprehensive RAGAS evaluation integration. This implementation provides multiple conversation formatting utilities specifically designed for different RAGAS metrics.

### 🎯 Key Features

- **📋 Multi-Format Conversation Export**: Three specialized functions for different RAGAS metrics
- **🧵 Thread Management**: Advanced conversation threading for parallel testing
- **📊 Interactive Commands**: Built-in commands for RAGAS sample generation
- **🔄 Real-time Formatting**: Live conversation formatting with visual output
- **⚡ Zero Simulation**: 100% authentic AI agent responses and tool execution

### 🏗️ Architecture Overview

```mermaid
graph TB
    subgraph "Production Agent Architecture"
        A[User Input] --> B[stream_graph_updates]
        B --> C[LangGraph Agent]
        C --> D{Tool Required?}
        D -->|Yes| E[Tavily Search]
        D -->|No| F[Direct Response]
        E --> G[ToolMessage]
        G --> F
        F --> H[Formatted Output]
        
        subgraph "RAGAS Integration Layer"
            I[get_conversation_for_ragas]
            J[get_conversation_for_tool_accuracy]
            K[get_conversation_for_goal_accuracy]
        end
        
        H --> I
        H --> J  
        H --> K
        
        I --> L[agent_topic_adherence]
        J --> M[_agent_tool_accuracy]
        K --> N[agent_goal_accuracy_with_reference]
        
        subgraph "Test Suite"
            O[test_real_agent.py]
            P[Comprehensive Logging]
            Q[Step-by-Step Process]
        end
        
        L --> O
        M --> O
        N --> O
        O --> P
        O --> Q
    end
    
    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style E fill:#fff3e0
    style H fill:#e8f5e8
    style I fill:#ffecb3
    style J fill:#ffecb3  
    style K fill:#ffecb3
    style O fill:#fce4ec
```

### 📦 RAGAS Conversation Formatters

The implementation provides three specialized functions for RAGAS metric evaluation:

#### 1. Topic Adherence Format
```python
def get_conversation_for_ragas(thread_id: str) -> List[LangChainMessage]:
    """
    Returns LangChain message objects for agent_topic_adherence metric.
    Usage: sample_input = get_conversation_for_ragas('thread_1')
    """
```

#### 2. Tool Accuracy Format  
```python
def get_conversation_for_tool_accuracy(thread_id: str) -> List[RagasMessage]:
    """
    Returns ragas message objects for ToolCallAccuracy metric.
    Usage: sample = get_conversation_for_tool_accuracy('thread_1')
    """
```

#### 3. Goal Accuracy Format
```python
def get_conversation_for_goal_accuracy(thread_id: str) -> MultiTurnSample:
    """
    Returns MultiTurnSample for agent_goal_accuracy_with_reference metric.
    Usage: sample = get_conversation_for_goal_accuracy('thread_1')
    """
```

### 💬 Interactive Commands

The agent provides interactive commands for RAGAS integration:

```bash
# Start the production agent
python src/4-final-agent-formated-response.py

# Interactive commands:
# 'ragas' - Get LangChain objects for agent_topic_adherence
# 'tool_accuracy' - Get ragas objects for _agent_tool_accuracy  
# 'goal_accuracy' - Get MultiTurnSample for agent_goal_accuracy_with_reference
# 'history' - Show formatted conversation history
# 'state' - Show current conversation state
# 'thread1/thread2' - Switch conversation threads
```

### 🧪 Integration with Test Suite

The production agent seamlessly integrates with the comprehensive test suite in `tests/test_real_agent.py`:

```python
# Tests directly use the production agent functions
from src import stream_graph_updates, get_conversation_for_tool_accuracy

# Execute real agent conversation
stream_graph_updates("What is the weather in Barcelona?", thread_id)

# Get RAGAS-formatted sample
conversation = get_conversation_for_tool_accuracy(thread_id)

# Evaluate with RAGAS
sample = MultiTurnSample(user_input=conversation)
scorer = TopicAdherenceScore(llm=evaluator_llm)
score = await scorer.multi_turn_ascore(sample)
```

### 🎭 Enhanced Test Logging

The test suite provides comprehensive step-by-step logging:

```
================================================================================
🧪 TEST: Topic Adherence Assessment (RAGAS)
================================================================================
🧵 Test Thread ID: topic_adherence_test_4287fc22

================================================================================
📋 STEP 1: FIRST AGENT INTERACTION (Weather Topic)
================================================================================
💬 Question: What are the current weather conditions in Barcelona, Spain?
🎯 Expected: Agent should use web search tools to get real weather data

[Real agent interaction with tool calls...]

================================================================================
📋 STEP 3: PREPARING RAGAS EVALUATION SAMPLE
================================================================================
📦 FINAL SAMPLE FOR RAGAS EVALUATION:
   • Message Count: 8
   • Reference Topics: 19 topics defined
   • Sample Type: MultiTurnSample for TopicAdherenceScore

🔍 WHAT IS BEING MEASURED:
   📊 METRIC: Topic Adherence Score (Recall Mode)
   📝 PURPOSE: Measures how well the agent stays focused on relevant topics
   🎯 EVALUATION: Checks if agent responses relate to defined reference topics
   📏 THRESHOLD: Score ≥ 0.4 indicates acceptable topic adherence

================================================================================
🔬 STEP 4: EXECUTING RAGAS EVALUATION
================================================================================
🎯 Topic Adherence Score: 1.000
📈 Score Interpretation: ⭐ EXCELLENT (≥0.8)
✅ TEST RESULT: PASS (EXCELLENT) - Topic adherence meets requirements
```

### 🎯 Zero Simulation Architecture

**Critical Feature**: The entire system uses **100% real data**:
- ✅ **Real LLM**: Qwen 2.5:7b-instruct via Ollama localhost:11434
- ✅ **Real Web Search**: Tavily API with live internet access  
- ✅ **Real Tool Execution**: Authentic tool calls and responses
- ✅ **Real Token Usage**: Actual computational costs
- ✅ **Real Timing**: Genuine API latencies and execution times
- 🚫 **Zero Mocks**: No simulated, fake, or hardcoded responses

This ensures that RAGAS evaluations reflect genuine agent performance in production scenarios.

## 🧪 RAGAS Agent Evaluation Framework

### Why RAGAS for AI Agent Testing?

**RAGAS (Retrieval Augmented Generation Assessment)** is the industry standard for evaluating AI agents because:

- **📊 Objective Metrics**: Provides quantifiable scores (0.0-1.0) instead of subjective assessments
- **🏭 Production Ready**: Used by major AI companies for agent evaluation in production
- **🔬 Multi-Dimensional**: Evaluates different aspects of agent behavior simultaneously
- **📈 Benchmarking**: Enables consistent comparison across agent versions and competitors
- **🤖 LLM-as-Judge**: Uses advanced LLMs to evaluate responses, mimicking human judgment at scale

### Test Architecture Overview

```mermaid
graph TD
    A["🧪 pytest test_real_agent.py"] --> B["🔧 RealAgentRunner"]
    B --> C["📦 Recreate Actual Agent"]
    C --> D["3_basic_chat_bot_with_tools_memory.py"]
    D --> E["💬 Execute Real Conversations"]
    E --> F["📊 RAGAS Evaluation"]
    
    subgraph "Agent Components"
        G["🤖 Ollama LLM<br/>qwen2.5:7b-instruct"]
        H["🔍 Tavily Search Tool"]
        I["🧠 InMemorySaver<br/>Conversation Memory"]
        J["🔄 LangGraph Orchestration"]
    end
    
    subgraph "RAGAS Metrics"
        K["📌 Topic Adherence Score<br/>Threshold: ≥0.4"]
        L["🛠️ Tool Usage Analysis<br/>Custom Validation"]
        M["🎯 Goal Achievement Score<br/>Threshold: ≥0.4"]
    end
    
    C --> G
    C --> H
    C --> I  
    C --> J
    F --> K
    F --> L
    F --> M
```

## 📊 RAGAS Metrics Deep Dive

### 1. 📌 Topic Adherence Score (TopicAdherenceScore)

#### What It Measures
Evaluates how well the agent maintains focus on appropriate topics and handles topic transitions professionally.

#### How It Works
```mermaid
graph LR
    A["👤 User Questions"] --> B["🤖 Agent Responses"]
    B --> C["📝 Conversation Sample"]
    C --> D["🏷️ Reference Topics<br/>weather, automation testing,<br/>CI/CD, technical info"]
    D --> E["🧠 RAGAS LLM Judge<br/>Ollama qwen2.5:7b"]
    E --> F["📊 Score: 0.0-1.0"]
    
    subgraph "Evaluation Process"
        G["1. Extract conversation turns"]
        H["2. Compare against reference topics"]
        I["3. Assess topic consistency"]
        J["4. Evaluate redirections"]
    end
    
    C --> G
    G --> H
    H --> I
    I --> J
    J --> E
```

#### Test Implementation
```python
# Real test from test_real_agent.py (lines 75-84)
sample = MultiTurnSample(
    user_input=conversation,
    reference_topics=[
        "weather information", "current weather conditions", "temperature", "climate",
        "automation testing", "automated testing", "test automation", "testing frameworks", 
        "quality assurance", "QA", "software testing", "testing best practices",
        "CI/CD pipelines", "continuous integration", "continuous deployment", "DevOps",
        "software development", "testing tools", "technical information"
    ]
)

scorer = TopicAdherenceScore(llm=langchain_llm_ragas_wrapper, mode="recall")
score = await scorer.multi_turn_ascore(sample)
```

#### Score Interpretation & Action Items

| Score Range | Interpretation | Action Items |
|-------------|----------------|--------------|
| **1.0** | Perfect topic adherence | ✅ Maintain current prompts and guardrails |
| **0.8-0.9** | Excellent focus | ✅ Minor prompt optimization opportunities |
| **0.6-0.7** | Good with minor drift | 🔄 Review topic boundaries in system prompts |
| **0.4-0.5** | Acceptable (passes threshold) | ⚠️ Strengthen topic filtering logic |
| **0.2-0.3** | Poor topic management | 🚨 Redesign agent instructions with explicit topic boundaries |
| **0.0-0.1** | Significant topic drift | 🚨 Complete prompt redesign required |

**Real Test Example:**
```bash
📊 RAGAS EVALUATION RESULTS:
   🎯 Adherence Score: 1.000
   📏 Acceptance Threshold: 0.4
   ✅ PASS: Acceptable topic adherence maintained
```

### 2. 🛠️ Tool Usage Analysis (Custom Implementation)

#### What It Measures  
Evaluates the agent's ability to correctly identify when tools are needed and use them with appropriate parameters.

#### How It Works
```mermaid
graph TD
    A["❓ User Query<br/>'Search recent testing news'"] --> B["🤖 Agent Analysis"]
    B --> C{"🤔 Tool Required?"}
    C -->|Yes| D["🔍 Tool Selection"]
    C -->|No| E["❌ FAIL: Should use tools"]
    D --> F["📝 Parameter Generation"]
    F --> G["🔧 Tool Execution"]
    G --> H["📊 Analysis Results"]
    
    subgraph "Validation Checks"
        I["✓ Tools activated > 0"]
        J["✓ Correct tool selected (Tavily)"]
        K["✓ Relevant search queries"]
        L["✓ Appropriate parameters"]
    end
    
    H --> I
    H --> J
    H --> K
    H --> L
```

#### Test Implementation
```python
# Real test from test_real_agent.py (lines 132-168)
research_question = "Please search for recent news about automation testing frameworks and tools"
result = agent.ask_agent(research_question)

# Validate tool usage
assert result['tools_used'] > 0, f"Agent should use tools for search requests but used {result['tools_used']} tools"

# Verify tool selection
tool_names = [tc['name'] for tc in result['tool_calls']]
if any("tavily" in name.lower() for name in tool_names):
    print(f"   ✅ Correctly selected Tavily for web search")

# Check query relevance
relevant_keywords = ["automation testing", "test automation", "testing frameworks", "selenium", "cypress", "playwright"]
query_text = ' '.join(queries).lower()
if any(keyword in query_text for keyword in relevant_keywords):
    print(f"   ✅ Search queries contain relevant keywords")
```

#### Analysis & Action Items

| Tool Usage Result | Interpretation | Action Items |
|-------------------|----------------|--------------|
| **Tools Used: >0, Correct Tool, Relevant Query** | Perfect tool usage | ✅ Tool integration working optimally |
| **Tools Used: >0, Correct Tool, Poor Query** | Tool selection good, query needs work | 🔄 Improve query generation logic |
| **Tools Used: >0, Wrong Tool** | Tool selection logic issues | ⚠️ Review tool routing conditions |
| **Tools Used: 0** | Failed to identify tool need | 🚨 Enhance tool usage detection |

**Real Test Example:**
```bash
📊 TOOL USAGE ANALYSIS:
   • Tools Activated: 1
   • Tool Call 1: tavily_search
     Arguments: {'query': 'recent news about automation testing frameworks and tools', 'search_depth': 'advanced'}
   ✅ Correctly selected Tavily for web search
   📝 Search Queries Generated: ['recent news about automation testing frameworks and tools']
   ✅ Search queries contain relevant keywords
```

### 3. 🎯 Goal Achievement Score (AgentGoalAccuracyWithReference)

#### What It Measures
Assesses how effectively the agent understands and fulfills complex user objectives using a reference standard.

#### How It Works
```mermaid
graph LR
    A["📋 User Goal<br/>'Research test automation<br/>frameworks and summarize'"] --> B["🤖 Agent Execution"]
    B --> C["🔍 Tool Usage"]
    C --> D["📝 Response Generation"] 
    D --> E["📚 Reference Standard<br/>'Agent should research and<br/>provide summary about test<br/>automation developments'"]
    E --> F["🧠 RAGAS LLM Judge"]
    F --> G["📊 Achievement Score<br/>0.0-1.0"]
    
    subgraph "Evaluation Criteria"
        H["Completeness of research"]
        I["Quality of summary"]
        J["Relevance to request"]
        K["Use of appropriate tools"]
    end
    
    F --> H
    F --> I  
    F --> J
    F --> K
```

#### Test Implementation
```python
# Real test from test_real_agent.py (lines 195-266)
task_request = "Please research the latest developments in test automation frameworks and provide a brief summary"

# Execute task
result = agent.ask_agent(task_request)

# Create conversation for RAGAS
conversation = [
    HumanMessage(content=task_request),
    AIMessage(
        content=result['response'], 
        tool_calls=[ToolCall(name=tc['name'], args=tc['args']) for tc in result['tool_calls']]
    )
]

# Define reference standard
reference_goal = "Agent should research and provide a summary about test automation framework developments"

# Evaluate with RAGAS
sample = MultiTurnSample(user_input=conversation, reference=reference_goal)
scorer = AgentGoalAccuracyWithReference(llm=langchain_llm_ragas_wrapper)
score = await scorer.multi_turn_ascore(sample)
```

#### Score Interpretation & Action Items

| Score Range | Performance Level | Action Items |
|-------------|------------------|--------------|
| **0.9-1.0** | Exceptional goal achievement | ✅ Document approach for consistency |
| **0.7-0.8** | Excellent goal fulfillment | ✅ Minor response quality improvements |
| **0.5-0.6** | Good goal understanding | 🔄 Enhance comprehensiveness of responses |
| **0.4** | Meets threshold | ⚠️ Improve goal parsing and execution logic |
| **0.2-0.3** | Poor goal achievement | 🚨 Redesign intent recognition system |
| **0.0-0.1** | Failed to understand goals | 🚨 Complete architecture review needed |

**Real Test Example:**
```bash
📊 GOAL ACHIEVEMENT RESULTS:
   🎯 Goal Accuracy Score: 1.000
   📋 Reference Standard: Agent should research and provide a summary about test automation framework developments
   🔧 Tools Used: 1
   📝 Response Length: 1747 characters
   ✅ EXCELLENT: High goal achievement accuracy
```

## 🏗️ Test Infrastructure Deep Dive

### RealAgentRunner Architecture

The `RealAgentRunner` class creates an exact replica of the production agent for testing:

```mermaid
graph TD
    A["🧪 Test Case"] --> B["🔧 RealAgentRunner(test_name)"]
    B --> C["📦 Agent Recreation"]
    
    subgraph "Agent Components Recreation"
        D["🤖 ChatOllama<br/>model: qwen2.5:7b-instruct<br/>temperature: 0.0<br/>base_url: localhost:11434"]
        E["🔍 Tavily Search Tool<br/>max_results: 2"]
        F["🧠 InMemorySaver<br/>Memory System"]
        G["🔄 StateGraph<br/>+ ToolNode<br/>+ tools_condition"]
    end
    
    subgraph "Test Execution"
        H["❓ ask_agent(question)"]
        I["💬 Agent Streaming"]
        J["📊 Response Capture"]
        K["🔧 Tool Call Analysis"]
    end
    
    C --> D
    C --> E
    C --> F
    C --> G
    B --> H
    H --> I
    I --> J
    I --> K
```

### Test File Structure

```
tests/
├── 📄 conftest.py                 # Test configuration & fixtures
├── 📄 test_real_agent.py          # 🎯 Main RAGAS evaluation suite
├── 📁 helpers/
│   ├── 📄 __init__.py
│   └── 📄 utils.py                # 🔧 RealAgentRunner implementation
└── 📄 pytest.ini                 # Pytest configuration
```

### Configuration Details

#### conftest.py - Test Environment Setup
- **Ollama Integration**: Connects to local Ollama server
- **Health Checks**: Validates server availability before tests
- **RAGAS LLM Wrapper**: Configures evaluation LLM
- **Temperature**: Set to 0.0 for deterministic responses

#### utils.py - Agent Recreation Logic  
- **Exact Replication**: Mirrors production agent setup exactly
- **Memory Management**: Uses InMemorySaver for conversation persistence
- **Tool Integration**: Configures Tavily search with same parameters
- **Response Capture**: Extracts structured data for RAGAS evaluation

## 🚀 Running the Evaluation Suite

### Prerequisites Setup

```bash
# 1. Create virtual environment
python -m venv venv_foundational_agents
source venv_foundational_agents/bin/activate  # Linux/Mac
# OR
venv_foundational_agents\Scripts\activate     # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama server with required model
ollama serve
ollama pull qwen2.5:7b-instruct

# 4. Optional: Configure API keys in .env
cp env.example .env
# Edit .env with your Tavily API key
```

### Test Execution Commands

#### Complete Evaluation Suite
```bash
# Run all RAGAS tests with detailed output
pytest tests/test_real_agent.py -v -s

# Expected runtime: ~3-5 minutes per test
# Total suite: ~10-15 minutes
```

#### Individual Test Execution  
```bash
# Topic Adherence Test
pytest tests/test_real_agent.py::test_real_agent_topic_adherence_simple -v -s

# Tool Usage Analysis  
pytest tests/test_real_agent.py::test_real_agent_tool_accuracy_simple -v -s

# Goal Achievement Test
pytest tests/test_real_agent.py::test_real_agent_goal_accuracy_with_reference -v -s
```

### 📝 Real Test Output Examples

**Topic Adherence Test Output:**
```bash
============================================================
🧪 TEST: Topic Adherence Assessment (RAGAS)  
============================================================
🔧 Inicializando agente real para: TopicAdherenceTest
✅ Loaded environment variables from /path/to/.env
🔍 LangSmith tracing enabled for project: langgraphagents
🔍 Tavily API key configured: tvly-dev...
✅ Tavily search tool configured (max_results=2)
✅ Tavily tool ready for web search
✅ Agente real listo

❓ Usuario: What are the current weather conditions in Barcelona, Spain?
🤖 Agente: The current weather conditions in Barcelona, Spain are as follows:
- Temperature: 26.3°C (79.3°F)
- Conditions: Patchy rain nearby...
🔧 Herramientas usadas: ['tavily_search']

❓ Usuario: What are the best practices for implementing automated testing in CI/CD pipelines?
🤖 Agente: Here are some best practices for implementing automated testing in CI/CD pipelines:
### 1. **Integrate Automated Testing...
🔧 Herramientas usadas: ['tavily_search']

📝 Constructing RAGAS conversation sample...
🎯 Evaluating topic adherence with RAGAS scorer...

📊 RAGAS EVALUATION RESULTS:
   🎯 Adherence Score: 1.000
   📏 Acceptance Threshold: 0.4
   ✅ PASS: Acceptable topic adherence maintained
✅ TEST COMPLETED: Topic adherence successfully evaluated
```

## 🎯 QA Best Practices & Action Items

### Performance Benchmarks

| Metric | Excellent | Good | Needs Improvement | Critical |
|--------|-----------|------|-------------------|----------|
| **Topic Adherence** | ≥0.8 | 0.6-0.7 | 0.4-0.5 | <0.4 |
| **Tool Usage** | 100% success | Minor issues | Partial success | Failure |
| **Goal Achievement** | ≥0.8 | 0.6-0.7 | 0.4-0.5 | <0.4 |

### Continuous Improvement Workflow

```mermaid
graph LR
    A["📊 Run RAGAS Tests"] --> B{"📈 Scores Meet<br/>Thresholds?"}
    B -->|Yes| C["✅ Deploy to Production"]
    B -->|No| D["📝 Analyze Failures"]
    D --> E["🔧 Implement Fixes"]
    E --> F["🧪 Retest"]
    F --> A
    
    subgraph "Fix Categories"
        G["📌 Topic: Update prompts/guardrails"]
        H["🛠️ Tools: Fix routing logic"]
        I["🎯 Goals: Enhance comprehension"]
    end
    
    D --> G
    D --> H
    D --> I
```

### Integration with CI/CD

```yaml
# Example GitHub Actions integration
name: RAGAS Agent Evaluation
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      ollama:
        image: ollama/ollama:latest
        ports:
          - 11434:11434
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Pull Ollama model
        run: ollama pull qwen2.5:7b-instruct
      - name: Run RAGAS evaluation
        run: pytest tests/test_real_agent.py -v
        env:
          TAVILY_API_KEY: ${{ secrets.TAVILY_API_KEY }}
```

## 📖 Quick Start Guide

### For QA Teams

1. **Install & Configure**
   ```bash
   git clone <repository>
   cd langGraphAgents
   pip install -r requirements.txt
   ollama serve && ollama pull qwen2.5:7b-instruct
   cp env.example .env  # Configure API keys
   ```

2. **🆕 Test Production Agent**
   ```bash
   # Interactive testing with RAGAS integration
   python src/4-final-agent-formated-response.py
   ```

3. **🆕 Run Comprehensive RAGAS Evaluation**
   ```bash
   # Complete evaluation suite with enhanced logging
   pytest tests/test_real_agent.py -v -s
   
   # Individual metric tests
   pytest tests/test_real_agent.py::test_real_agent_topic_adherence_simple -v -s
   pytest tests/test_real_agent.py::test_real_agent_tool_accuracy_simple -v -s  
   pytest tests/test_real_agent.py::test_real_agent_goal_accuracy_with_reference -v -s
   ```

4. **Interpret Enhanced Results**
   - **Step-by-Step Process**: Follow detailed logging from agent interaction to RAGAS evaluation
   - **Sample Inspection**: See exactly what data is sent to RAGAS for evaluation
   - **Metric Explanations**: Understand what each metric measures and why
   - **Score Interpretation**: Review detailed score analysis with actionable insights
   - **Threshold Compliance**: Check all metrics meet ≥0.4 threshold

5. **Implement Improvements**
   - **Topic Issues**: Update system prompts and reference topics
   - **Tool Issues**: Fix routing logic and tool selection accuracy
   - **Goal Issues**: Enhance task comprehension and completion strategies

6. **🆕 Validate Changes with Real Data**
   ```bash
   # Re-run with 100% real agent responses
   pytest tests/test_real_agent.py -v -s
   ```

### For Development Teams

1. **🆕 Production Agent Development** 
   - Follow tutorials 1-3 for foundational concepts
   - **Focus on `src/4-final-agent-formated-response.py`** as production implementation
   - Utilize built-in RAGAS formatting functions for seamless evaluation integration

2. **🆕 Enhanced Testing Integration**
   - Tests directly import and use your agent functions
   - **Zero mocking**: 100% real agent responses with authentic tool execution
   - **Multi-format support**: Automatic conversion for different RAGAS metrics
   - **Comprehensive logging**: Step-by-step process visibility

3. **🆕 Advanced Monitoring & Development**
   - **Interactive Development**: Use built-in commands (ragas, tool_accuracy, goal_accuracy)
   - **Real-time Validation**: Test conversations instantly with RAGAS formatting
   - **Thread Management**: Parallel testing with isolated conversation contexts
   - **Performance Tracking**: Monitor real token usage, timing, and API costs

4. **Integration Workflow**
   ```python
   # Development workflow example
   from src.4-final-agent-formated-response import stream_graph_updates, get_conversation_for_tool_accuracy
   
   # Test your agent
   stream_graph_updates("Your test question", "dev_thread")
   
   # Get RAGAS sample for evaluation
   sample = get_conversation_for_tool_accuracy("dev_thread")
   
   # Ready for RAGAS evaluation
   ```

## 🔧 Advanced Configuration

### Custom Thresholds

Modify thresholds in `test_real_agent.py`:

```python
# Topic Adherence (line 105)
assert score >= 0.6, f"Custom threshold: RAGAS TopicAdherenceScore {score:.3f} below 0.6"

# Goal Achievement (line 265)  
assert score >= 0.7, f"Custom threshold: RAGAS AgentGoalAccuracyWithReference score {score:.3f} below 0.7"
```

### Different LLM Backends

Update `conftest.py` for alternative evaluation LLMs:

```python
# Using OpenAI for evaluation
llm = ChatOpenAI(model="gpt-4", temperature=0.0)
return LangchainLLMWrapper(llm)

# Using Anthropic for evaluation  
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.0)
return LangchainLLMWrapper(llm)
```

### Additional Test Scenarios

Extend `test_real_agent.py` with custom scenarios:

```python
@pytest.mark.asyncio
async def test_custom_scenario(langchain_llm_ragas_wrapper):
    agent = RealAgentRunner("CustomTest")
    
    # Your custom test logic here
    result = agent.ask_agent("Your custom query")
    
    # Custom RAGAS evaluation
    # Implementation details...
```

## 📚 Learning Resources & References

### RAGAS Documentation
- [RAGAS Official Documentation](https://docs.ragas.io/)
- [RAGAS GitHub Repository](https://github.com/explodinggradients/ragas)
- [Multi-turn Evaluation Guide](https://docs.ragas.io/en/stable/concepts/metrics/multi_turn.html)

### LangGraph Resources
- [LangGraph Concepts](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [LangGraph Testing Best Practices](https://langchain-ai.github.io/langgraph/how-tos/testing/)

### QA & Testing
- [AI Agent Testing Methodologies](https://blog.langchain.dev/testing-ai-agents/)
- [LLM Evaluation Best Practices](https://blog.langchain.dev/evaluating-llm-apps/)
- [Production AI System Monitoring](https://docs.smith.langchain.com/)

## 🤝 Contributing

This project follows QA best practices for AI agent testing. Contributions that enhance testing coverage, improve evaluation accuracy, or add new RAGAS metrics are welcome.

### Development Guidelines
- All agent modifications must pass RAGAS evaluation  
- New features require corresponding test coverage
- Performance regressions (score decreases) must be justified
- Documentation updates required for new metrics or thresholds

## 📄 License

This project is for educational and professional development purposes, following the MIT License.
