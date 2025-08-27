"""
Test Utilities for Real Agent Evaluation

This module contains utilities for running and interacting with the actual
LangGraph agent for testing purposes.
"""

import sys
import os
import uuid
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import agent dependencies
from src.utils.langchain_setup import setup_langsmith
from src.utils.tavily_setup import setup_tavily
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict
from typing import Annotated


class State(TypedDict):
    """State schema for the LangGraph agent"""
    messages: Annotated[list, add_messages]


class RealAgentRunner:
    """
    Runs the actual LangGraph agent and captures responses for RAGAS evaluation.
    
    This class creates and manages a real instance of the agent defined in
    3_basic_chat_bot_with_tools_memory.py for testing purposes.
    
    Each test can provide a custom name for LangSmith tracking, making it easy
    to identify individual test runs in the LangSmith dashboard.
    """
    
    def __init__(self, test_name: str = "RealAgentTest"):
        """
        Initialize the real agent with memory and tools
        
        Args:
            test_name (str): Custom name for LangSmith tracking (default: "RealAgentTest")
        """
        print(f"🔧 Inicializando agente real para: {test_name}")
        
        # Store test name for LangSmith tracking
        self.test_name = test_name
        
        # Initialize memory for conversation persistence
        self.memory = InMemorySaver()
        
        # Create the complete agent graph
        self.agent = self._create_agent()
        
        print("✅ Agente real listo")
    
    def _create_agent(self):
        """
        Create the actual agent from the original script.
        
        This method replicates the agent creation logic from
        3_basic_chat_bot_with_tools_memory.py to ensure we're testing
        the real implementation.
        
        Returns:
            Compiled LangGraph agent with tools and memory
        """
        
        # Setup LangSmith tracking with custom test name (optional, fails gracefully)
        try:
            # Temporarily set the project name for this test
            original_project = os.environ.get("LANGCHAIN_PROJECT")
            os.environ["LANGCHAIN_PROJECT"] = self.test_name
            
            setup_langsmith()
            
            # Restore original project name if it existed
            if original_project:
                os.environ["LANGCHAIN_PROJECT"] = original_project
            else:
                os.environ.pop("LANGCHAIN_PROJECT", None)
                
        except Exception as e:
            print(f"⚠️  LangSmith setup failed: {e}")
            pass
        
        # Setup Tavily search tool for web searches
        tool = setup_tavily(max_results=2)
        tools = [tool] if tool else []
        
        # Initialize LLM with consistent parameters
        llm = ChatOllama(
            model="qwen2.5:7b-instruct",
            base_url="http://localhost:11434",
            temperature=0.0,  # Deterministic responses for testing
        )
        
        # Create the state graph builder
        graph_builder = StateGraph(State)
        
        # Configure agent with or without tools
        if tools:
            # Create LLM with tool binding for function calling
            llm_with_tools = llm.bind_tools(tools)
            
            def chatbot(state: State):
                """Main chatbot node with tool calling capability"""
                return {"messages": [llm_with_tools.invoke(state["messages"])]}
            
            # Add main chatbot node
            graph_builder.add_node("chatbot", chatbot)
            
            # Add tool execution node
            tool_node = ToolNode(tools=tools)
            graph_builder.add_node("tools", tool_node)
            
            # Add conditional edges for tool calling logic
            graph_builder.add_conditional_edges(
                "chatbot",
                tools_condition,  # Decides whether to use tools or end
            )
            
            # Add edges for conversation flow
            graph_builder.add_edge("tools", "chatbot")  # After tool use, return to chatbot
            graph_builder.add_edge(START, "chatbot")   # Start with chatbot
            
        else:
            # Fallback to basic chatbot without tools
            def chatbot(state: State):
                """Basic chatbot node without tool capabilities"""
                return {"messages": [llm.invoke(state["messages"])]}
            
            graph_builder.add_node("chatbot", chatbot)
            graph_builder.add_edge(START, "chatbot")
            graph_builder.add_edge("chatbot", END)
        
        # Compile the graph with memory checkpoint for conversation persistence
        return graph_builder.compile(checkpointer=self.memory)
    
    def ask_agent(self, question: str, thread_id: str = None) -> dict:
        """
        Ask the agent a single question and capture structured response data.
        
        This method interacts with the real agent and extracts all relevant
        information needed for RAGAS evaluation including tool usage.
        
        Args:
            question (str): The question to ask the agent
            thread_id (str, optional): Thread ID for conversation continuity.
                                     If None, generates a new UUID.
        
        Returns:
            dict: Structured response containing:
                - question: Original question asked
                - response: Agent's text response  
                - tools_used: Number of tools called
                - tool_calls: List of tool calls with names and arguments
        """
        # Generate unique thread ID if not provided
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        # Configure the agent with thread-specific memory
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"❓ Usuario: {question}")
        
        # Execute agent with streaming to capture all events
        events = self.agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config,
            stream_mode="values",  # Get state values at each step
        )
        
        # Initialize response tracking variables
        agent_response = ""
        tool_calls = []
        
        # Process all events from the agent execution
        for event in events:
            # Each event contains the current state
            if isinstance(event, dict) and "messages" in event:
                # Get the most recent message
                message = event["messages"][-1]
                
                # Extract text content from AI messages
                if hasattr(message, 'content') and message.content:
                    agent_response = message.content
                
                # Extract tool calls for analysis
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tc in message.tool_calls:
                        tool_calls.append({
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {})
                        })
        
        # Log response summary for debugging
        print(f"🤖 Agente: {agent_response[:150]}...")
        if tool_calls:
            print(f"🔧 Herramientas usadas: {[tc['name'] for tc in tool_calls]}")
        
        # Return structured data for evaluation
        return {
            "question": question,
            "response": agent_response,
            "tools_used": len(tool_calls),
            "tool_calls": tool_calls
        }
