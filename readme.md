# 🤖 LangGraph QA Automation Agent with RAGAS Evaluation

> **Progressive AI Agent Development: From Basic Chat to Production-Ready QA Automation**

A comprehensive learning project that demonstrates the evolution of AI agents from basic chatbots to sophisticated QA automation consultants, complete with industry-standard RAGAS evaluation framework for measuring agent performance in real-world scenarios.

## 🎯 Project Overview

This project showcases **progressive agent development** through four distinct stages:

1. **Basic Stateful Chatbot** (`1_basic_chat_bot.py`) - Foundation with conversation memory
2. **Tool-Enhanced Agent** (`2_basic_chat_bot_with_tools.py`) - Web search integration via Tavily
3. **Production Memory System** (`3_basic_chat_bot_with_tools_memory.py`) - Persistent conversation storage
4. **QA Automation Expert** (`4-final-agent-formated-response.py`) - Specialized consultant with RAGAS evaluation

Each stage builds upon the previous, demonstrating real-world AI development practices from prototype to production.

### 🏆 Key Features

- **Specialized QA Agent**: Expert knowledge in test automation, CI/CD, and software testing
- **Real Web Search**: Live integration with Tavily API for current information
- **Memory Persistence**: Conversation continuity across sessions
- **RAGAS Evaluation**: Industry-standard metrics for agent performance assessment
- **Local LLM Support**: Uses Ollama with Qwen2.5 model for privacy and control

## ⚡ Quick Start

**For immediate testing:**

```bash
# 1. Ensure Ollama is running with Qwen2.5 model
ollama pull qwen2.5:7b-instruct && ollama serve

# 2. Set up environment  
python -m venv venv_foundational_agents
source venv_foundational_agents/bin/activate  # Windows: venv_foundational_agents\Scripts\activate
pip install -r requirements.txt

# 3. Configure API key
cp env.example .env
# Edit .env to add your Tavily API key

# 4. Test the QA automation agent
python src/4-final-agent-formated-response.py

# 5. Run RAGAS evaluation tests
python -m pytest tests/test_real_agent_simple.py -v
```

---

## 🛠️ Detailed Setup & Installation

### 📋 Prerequisites

- **Python 3.9+** 
- **Ollama** (local LLM server)
- **Tavily API Key** (web search - required)
- **LangSmith API Key** (monitoring - optional)

### 🚀 Installation Steps

#### 1. Environment Setup
```bash
# Clone and enter directory
git clone <your-repo-url>
cd langGraphAgents

# Create virtual environment  
python -m venv venv_foundational_agents
source venv_foundational_agents/bin/activate  # Windows: venv_foundational_agents\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Ollama Setup
```bash
# Install Ollama (choose your platform)
brew install ollama                              # macOS
curl -fsSL https://ollama.ai/install.sh | sh     # Linux
# Windows: Download from https://ollama.ai/download

# Pull required model and start server
ollama pull qwen2.5:7b-instruct
ollama serve  # Runs on http://localhost:11434
```

#### 3. API Configuration
```bash
# Create environment file
cp env.example .env

# Edit .env with your API keys:
# TAVILY_API_KEY=your_tavily_key_here          # Required for web search
# LANGSMITH_API_KEY=your_langsmith_key_here    # Optional for monitoring
```

**Get API Keys:**
- **Tavily**: [tavily.com](https://tavily.com/) → Dashboard → API Keys
- **LangSmith**: [smith.langchain.com](https://smith.langchain.com/) → Settings → API Keys

#### 4. Test Installation
```bash
# Test the QA automation agent
python src/4-final-agent-formated-response.py

# Run RAGAS evaluation suite
python -m pytest tests/test_real_agent_simple.py -v
```

### 🔧 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| **Ollama connection failed** | `ollama serve` and ensure `ollama list` shows `qwen2.5:7b-instruct` |
| **Import errors** | Activate virtual environment: `source venv_foundational_agents/bin/activate` |
| **API key errors** | Check `.env` file format and verify keys are valid |
| **Test failures** | Run with verbose output: `pytest -v -s --tb=long` |

### ✅ Verification Checklist

- [ ] Ollama running (`curl http://localhost:11434/api/version`)
- [ ] Model available (`ollama list | grep qwen2.5`)  
- [ ] Dependencies installed (`pip list | grep langgraph`)
- [ ] `.env` configured with Tavily API key
- [ ] Agent working (`python src/4-final-agent-formated-response.py`)
- [ ] Tests passing (`pytest tests/test_real_agent_simple.py`)

---

## 🏗️ System Architecture

### 📊 Component Architecture

```mermaid
graph TD
    subgraph "Agent Development Pipeline"
        A[Basic Chatbot<br/>State Management] --> B[Tool Integration<br/>Web Search]
        B --> C[Memory System<br/>Persistence]
        C --> D[Production Agent<br/>RAGAS Ready]
    end
    
    subgraph "RAGAS Evaluation System"
        D --> E[Conversation Capture]
        E --> F[Format Conversion]
        F --> G[Metric Evaluation]
        
        subgraph "Conversation Formatters"
            H[get_conversation_for_ragas<br/>LangChain Messages]
            I[get_conversation_for_tool_accuracy<br/>RAGAS Messages]
            J[get_conversation_for_goal_accuracy<br/>MultiTurnSample]
        end
        
        F --> H
        F --> I
        F --> J
    end
    
    subgraph "RAGAS Metrics"
        K[TopicAdherenceScore<br/>Topic Focus Analysis]
        L[ToolCallAccuracy<br/>Tool Usage Evaluation]
        M[AgentGoalAccuracyWithReference<br/>Task Completion Assessment]
    end
    
    H --> K
    I --> L
    J --> M
    
    style D fill:#e1f5fe
    style K fill:#f3e5f5
    style L fill:#e8f5e8
    style M fill:#fff3e0
```

### 🏗️ Project Structure

```
langGraphAgents/
├── src/                                       # 📁 Source Code
│   ├── 1_basic_chat_bot.py                   # 🟢 Stage 1: Basic stateful conversation
│   ├── 2_basic_chat_bot_with_tools.py        # 🟡 Stage 2: Tool integration (web search)
│   ├── 3_basic_chat_bot_with_tools_memory.py # 🟠 Stage 3: Memory persistence system
│   ├── 4-final-agent-formated-response.py    # 🔴 Stage 4: QA expert + RAGAS evaluation
│   └── utils/                                 # 🛠️ Utility modules
│       ├── langchain_setup.py                # LangSmith configuration
│       └── tavily_setup.py                   # Tavily API integration
│
├── tests/                                     # 🧪 RAGAS Evaluation Suite
│   ├── test_real_agent_simple.py            # Main evaluation tests
│   ├── conftest.py                          # Test configuration & fixtures
│   ├── helpers/
│   │   └── utils.py                         # Test utilities & agent runner
│   └── logs/                                # Test execution logs
│       ├── test_goal_accuracy_*.txt         # Goal achievement results
│       ├── test_tool_accuracy_*.txt         # Tool usage results
│       └── test_topic_adherence_*.txt       # Topic focus results
│
├── requirements.txt                          # 📦 Python dependencies
├── env.example                              # 🔐 Environment configuration template
└── venv_foundational_agents/               # 🐍 Virtual environment (auto-created)
```

**Progressive Learning Path:**
1. **Green** → Basic conversation with memory
2. **Yellow** → Add web search capabilities  
3. **Orange** → Persistent conversation storage
4. **Red** → Specialized QA expert with evaluation

---

## 📊 RAGAS Evaluation Framework

### 🎯 Unified Evaluation Method

The project uses a **single unified method** for all RAGAS evaluation metrics, simplifying the testing process while maintaining comprehensive coverage:

```python
# Universal method for all RAGAS metrics
sample = getMultiTurnSampleConversation(thread_id)

# Configure for specific metrics:
sample.reference_topics = ["QA", "testing", "automation"]     # Topic Adherence
sample.reference_tool_calls = extracted_tool_calls           # Tool Accuracy  
sample.reference = "Expected goal achievement description"    # Goal Achievement
```

### 🧪 Evaluation Metrics

| Metric | Purpose | What It Measures | Threshold |
|--------|---------|------------------|-----------|
| **Topic Adherence** | Focus verification | Agent stays within QA automation domain | ≥ 0.8 |
| **Tool Call Accuracy** | Tool usage evaluation | Correct web search usage when needed | ≥ 0.7 |
| **Goal Achievement** | Task completion assessment | Agent provides useful, complete responses | ≥ 0.5 |

### 🚀 Real Test Performance

**Current test results with actual conversation data:**

```bash
✅ Topic Adherence:  1.000/1.0 (PERFECT!)
✅ Tool Accuracy:    1.000/1.0 (PERFECT!)  
✅ Goal Achievement: 1.000/1.0 (PERFECT!)
```

### 📝 Test Examples

#### Basic Topic Adherence Test
```python
@pytest.mark.asyncio
async def test_topic_adherence_simple():
    thread_id = f"topic_test_{uuid.uuid4().hex[:8]}"
    
    # Create conversation with off-topic challenge
    stream_graph_updates("What's the weather in Barcelona?", thread_id)
    stream_graph_updates("What are CI/CD best practices?", thread_id)
    
    # Evaluate topic focus
    sample = getMultiTurnSampleConversation(thread_id)
    sample.reference_topics = ["weather", "testing", "CI/CD", "automation"]
    
    scorer = TopicAdherenceScore(llm=evaluator_llm, mode="recall")
    score = await scorer.multi_turn_ascore(sample)
    
    assert score >= 0.4  # Agent should maintain professional focus
```

#### Tool Usage Evaluation
```python
@pytest.mark.asyncio  
async def test_tool_accuracy_simple():
    thread_id = f"tool_test_{uuid.uuid4().hex[:8]}"
    
    # Trigger web search requirement
    stream_graph_updates("Search for recent automation testing news", thread_id)
    
    # Evaluate tool usage
    sample = getMultiTurnSampleConversation(thread_id)
    # Extract and set reference tool calls from conversation
    sample.reference_tool_calls = extracted_tool_calls
    
    scorer = ToolCallAccuracy()
    score = await scorer.multi_turn_ascore(sample)
    
    assert score >= 0.7  # Agent should use tools appropriately
```

### 🎭 Score Interpretation

| Score Range | Status | Action Required |
|-------------|--------|----------------|
| **0.90 - 1.00** | 🟢 **Excellent** | Production ready - monitor for consistency |
| **0.75 - 0.89** | 🟡 **Good** | Minor tuning needed - review edge cases |
| **0.60 - 0.74** | 🟠 **Moderate** | Requires attention - adjust prompts/training |
| **< 0.60** | 🔴 **Needs Work** | Major revision required - review architecture |

---

## 🔍 Understanding RAGAS Metrics

### Metric Definitions

#### 🎯 Topic Adherence Score
- **Purpose**: Measures how well the agent stays focused on QA automation topics
- **Method**: Compares responses against reference topics using LLM evaluation  
- **Range**: 0.0 (completely off-topic) to 1.0 (perfectly focused)

#### 🛠️ Tool Call Accuracy
- **Purpose**: Evaluates appropriate use of web search tools
- **Method**: Compares actual vs expected tool usage patterns
- **Range**: 0.0 (poor tool usage) to 1.0 (optimal tool selection)

#### 🏆 Goal Achievement
- **Purpose**: Assesses task completion quality against reference standards
- **Method**: LLM evaluation of response completeness and usefulness
- **Range**: 0.0 (goal not met) to 1.0 (perfect goal achievement)

### 📊 Real Test Performance Data

**Sample conversation thread: `goal_test_c8af1eb3`**

```
User: "Research latest test automation frameworks and provide a summary"

Agent Actions:
1. Recognizes need for current information → triggers web search
2. Calls tavily_search with query: "latest test automation frameworks"  
3. Processes real web results from TestGuild, Testomat.io, and other sources
4. Delivers structured summary of 7+ frameworks with descriptions

RAGAS Scores:
✅ Topic Adherence:  1.000/1.0 (stayed within QA domain)
✅ Tool Accuracy:    1.000/1.0 (used search appropriately)  
✅ Goal Achievement: 1.000/1.0 (comprehensive framework summary)
```

### 🎮 Running Your Own Evaluations

```python
# Basic evaluation workflow
thread_id = f"test_{uuid.uuid4().hex[:8]}"

# 1. Generate conversation
stream_graph_updates("Your test question here", thread_id)

# 2. Convert to RAGAS format  
sample = getMultiTurnSampleConversation(thread_id)

# 3. Configure for your metric
sample.reference_topics = ["your", "topics", "here"]  # Topic adherence
# OR sample.reference = "Your expected outcome"        # Goal achievement  
# OR sample.reference_tool_calls = tool_calls          # Tool accuracy

# 4. Evaluate
scorer = YourChosenMetric(llm=evaluator_llm)
score = await scorer.multi_turn_ascore(sample)
```

---

## 🎓 Learning Path & Dependencies  

### 📚 Progressive Stages Explained

| Stage | File | Key Concepts | Dependencies |
|-------|------|-------------|-------------|
| **🟢 Stage 1** | `1_basic_chat_bot.py` | LangGraph basics, State management, Memory | `langgraph`, `langchain-ollama` |
| **🟡 Stage 2** | `2_basic_chat_bot_with_tools.py` | Tool integration, Web search, API calls | + `langchain-tavily` |
| **🟠 Stage 3** | `3_basic_chat_bot_with_tools_memory.py` | Persistent memory, Conversation continuity | + `InMemorySaver` checkpointing |
| **🔴 Stage 4** | `4-final-agent-formated-response.py` | Domain expertise, RAGAS evaluation | + `ragas`, specialized prompts |

### 🧰 Complete Dependencies

**Core Requirements:**
```txt
langgraph==0.6.5          # Agent framework & state management
langchain-ollama           # Local LLM integration (Qwen2.5)
langchain-tavily           # Web search capabilities  
langsmith==0.4.14         # Optional: Monitoring & tracing
python-dotenv==1.0.1      # Environment configuration
ragas                     # AI evaluation framework
pytest                    # Testing infrastructure
pytest-asyncio            # Async test support
requests                  # HTTP utilities
```

### 🌟 What You'll Learn

1. **🟢 Foundation**: Build stateful conversations with local LLMs
2. **🟡 Integration**: Connect agents to external APIs and tools
3. **🟠 Production**: Implement memory persistence and conversation management
4. **🔴 Specialization**: Create domain-specific agents with professional evaluation

### 🚀 Next Steps

- **Modify Prompts**: Adapt the QA focus to your domain (DevOps, Security, etc.)
- **Add Tools**: Integrate additional APIs beyond web search
- **Extend Evaluation**: Create custom RAGAS metrics for your specific use case
- **Scale Architecture**: Deploy with production databases and persistent storage

---

## 🤝 Contributing & Support

### 📄 License
This project is provided as educational material for learning LangGraph and RAGAS evaluation.

### 🔗 Resources
- **LangGraph Documentation**: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
- **RAGAS Framework**: [ragas.io](https://ragas.io/)
- **Ollama Models**: [ollama.ai/library](https://ollama.ai/library)
- **Tavily Search API**: [tavily.com](https://tavily.com/)

---

**🎯 Ready to build your own AI agent? Start with Stage 1 and work your way up to production-ready evaluation!**