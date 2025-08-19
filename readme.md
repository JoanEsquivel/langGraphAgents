# Project to learn to build agents using Langchain & LangGraph

LangChain provides the building blocks (LLMs, prompts, retrievers, agents) to create AI applications.

LangGraph builds on top of LangChain to orchestrate those components as a graph, enabling stateful, multi-step, and controllable agent workflows.

## Build a basic agent
[Introduction](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)

[Build a basic chatbol](https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/)

## Setup

Create a virtual environment

```
pyenv install 3.11

pyenv shell 3.11

python -m venv venv_foundational_agents

source venv_foundational_agents/bin/activate
```

Install the requirements

```
pip install -r requirements.txt
```

## Basic Chatbot Explanation

The `src/basic_chat_bot.py` file implements a basic chatbot using LangGraph that connects to a local Ollama model. Below is a step-by-step explanation of how it works:

### Main Components

#### 1. Imports and model configuration
```python
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

llm = ChatOllama(
    model="qwen2.5:7b-instruct",
    base_url="http://localhost:11434",
    temperature=0.0,
)
```
- **ChatOllama**: Client to connect to the local Ollama server
- **StateGraph**: Defines the chatbot structure as a state machine
- **add_messages**: Pre-built function to handle message accumulation

#### 2. State definition
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```
- **State**: Structure that maintains the conversation history
- **add_messages**: Ensures new messages are added to the history instead of overwriting it

#### 3. Graph construction
```python
graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()
```
- **chatbot()**: Function that processes the current state and generates an LLM response
- **add_node()**: Adds the chatbot node to the graph
- **add_edge()**: Defines the flow: START → chatbot → END
- **compile()**: Compiles the graph to make it executable

#### 4. Streaming function
```python
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
```
- **stream()**: Executes the graph and streams results in real-time
- Takes user input and processes it through the graph
- Prints the assistant's response

#### 5. Main interaction loop
```python
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback for environments without input() available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
```
- Infinite loop that requests user input
- Handles exit commands ("quit", "exit", "q")
- Includes a fallback for environments where `input()` is not available

### Execution Flow

1. **Start**: User enters a message
2. **Processing**: The message is sent to the graph as initial state
3. **LLM**: The chatbot node invokes the Ollama model with the message history
4. **Response**: The model generates a response that gets added to the state
5. **Output**: The response is displayed to the user
6. **Repeat**: The cycle continues until the user enters an exit command

### Running the Chatbot

To run the chatbot:

```bash
# Activate the virtual environment
source venv_foundational_agents/bin/activate

# Run the chatbot
python src/basic_chat_bot.py
```

**Prerequisites:**
- Ollama must be running on `localhost:11434`
- The `qwen2.5:7b-instruct` model must be downloaded in Ollama