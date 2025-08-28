"""
Simplified QA Automation Agent Evaluation Tests with RAGAS

These tests are optimized for compatibility with local RAGAS evaluation using Ollama models.
Each test is simplified to create manageable conversation complexity while maintaining 
comprehensive coverage of agent capabilities within the RAGAS framework.

Simplified Test Coverage:
- Topic Adherence: Agent stays within QA automation boundaries (2-turn conversation with off-topic challenge)
- Tool Call Accuracy: Agent uses search tools correctly for focused research (4-message conversation with tool calls)  
- Goal Achievement: Agent provides accurate QA information (2-turn Q&A about API testing concepts)

All tests use your getMultiTurnSampleConversation() method and pure RAGAS metrics without fallbacks.
"""

import sys
import os
import pytest
import uuid
import importlib.util

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAGAS components
from ragas.dataset_schema import MultiTurnSample, ToolCall
from ragas.metrics import TopicAdherenceScore, ToolCallAccuracy, AgentGoalAccuracyWithReference

# Import agent utilities
from src.utils import setup_langsmith
setup_langsmith()

# Add src to path and import agent functions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import the agent module using importlib
spec = importlib.util.spec_from_file_location(
    "agent_final", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "4-final-agent-formated-response.py")
)
agent_final = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_final)

# Make functions available at module level
stream_graph_updates = agent_final.stream_graph_updates
getMultiTurnSampleConversation = agent_final.getMultiTurnSampleConversation

print("Agent functions imported successfully")


@pytest.mark.asyncio
async def test_topic_adherence_simple(langchain_llm_ragas_wrapper):
    """
    Topic Adherence Test - Simplified for RAGAS Compatibility
    
    Tests if the agent stays within QA automation boundaries.
    Uses focused questions that create manageable conversation complexity for RAGAS evaluation.
    """
    print("\nTOPIC ADHERENCE TEST - SIMPLIFIED QA AUTOMATION FOCUS")
    print("="*50)
    
    # Create conversation
    thread_id = f"topic_test_{uuid.uuid4().hex[:8]}"
    print(f"Thread: {thread_id}")
    
    # Ask about functional testing (focused, shorter question)
    print("\nQuestion 1: Functional Testing")
    stream_graph_updates("What is functional testing?", thread_id)
    
    # Test topic adherence with off-topic question
    print("\nQuestion 2: Off-Topic Challenge")
    stream_graph_updates("What's the weather today?", thread_id)
    
    # Get conversation using unified method
    print("\nGetting conversation using unified method...")
    sample = getMultiTurnSampleConversation(thread_id)
    print(f"Got MultiTurnSample with {len(sample.user_input)} messages")
    
    # Set focused QA automation reference topics
    reference_topics = [
        "functional testing", "software testing", "test automation", 
        "quality assurance", "testing strategies", "testing practices"
    ]
    sample.reference_topics = reference_topics
    
    # Evaluate with RAGAS
    print("\nRunning RAGAS evaluation...")
    scorer = TopicAdherenceScore(llm=langchain_llm_ragas_wrapper, mode="recall")
    score = await scorer.multi_turn_ascore(sample)
    
    print(f"\nResults:")
    print(f"   Score: {score:.3f}")
    print(f"   Threshold: >= 0.7")
    print(f"   Status: {'PASS' if score >= 0.7 else 'FAIL'}")
    
    assert score >= 0.7, f"Topic adherence score {score:.3f} below threshold"


@pytest.mark.asyncio  
async def test_tool_accuracy_simple(langchain_llm_ragas_wrapper):
    """
    Tool Call Accuracy Test - Simplified for RAGAS Compatibility
    
    Tests if agent uses search tools correctly for focused QA automation research.
    Uses a single, clear question that requires tool usage but keeps complexity manageable.
    """
    print("\nTOOL CALL ACCURACY TEST - SIMPLIFIED QA RESEARCH")
    print("="*50)
    
    # Create conversation
    thread_id = f"tool_test_{uuid.uuid4().hex[:8]}"
    print(f"Thread: {thread_id}")
    
    # Ask focused question that requires current research
    print("\nQuestion: Focused Research Task")
    stream_graph_updates(
        "Find current information about the newest API testing framework released in 2024",
        thread_id
    )
    
    # Get conversation using unified method
    print("\nGetting conversation using unified method...")
    sample = getMultiTurnSampleConversation(thread_id)
    print(f"Got MultiTurnSample with {len(sample.user_input)} messages")
    
    # Define expected tool calls for the simplified question
    # The test asks about: newest API testing framework released in 2024
    # A competent agent should make a search for current information
    expected_tool_calls = [
        ToolCall(
            name="tavily_search", 
            args={"query": "newest API testing framework 2024"}
        )
    ]
    
    # Set reference tool calls following RAGAS documentation
    sample.reference_tool_calls = expected_tool_calls
    
    # Evaluate with RAGAS
    print("\nRunning RAGAS evaluation...")
    scorer = ToolCallAccuracy()
    score = await scorer.multi_turn_ascore(sample)
    
    print(f"\nResults:")
    print(f"   Score: {score:.3f}")
    print(f"   Threshold: >= 0.7")
    print(f"   Status: {'PASS' if score >= 0.7 else 'FAIL'}")
    
    assert score >= 0.7, f"Tool accuracy score {score:.3f} below threshold"


@pytest.mark.asyncio
async def test_goal_accuracy_simple(langchain_llm_ragas_wrapper):
    """
    Goal Achievement Test - Simplified for RAGAS Compatibility
    
    Tests if agent achieves a focused QA automation goal.
    Uses a simple, clear task that creates manageable conversation complexity.
    """
    print("\nGOAL ACHIEVEMENT TEST - SIMPLIFIED QA FOCUS")
    print("="*50)
    
    # Create conversation
    thread_id = f"goal_simple_{uuid.uuid4().hex[:8]}"
    print(f"Thread: {thread_id}")
    
    # Give agent a simple, clear goal
    print("\nTask: Simple QA Concept Question")
    simple_task = "What is API testing and why is it important?"
    stream_graph_updates(simple_task, thread_id)
    
    # Get conversation using unified method
    print("\nGetting conversation using unified method...")
    sample = getMultiTurnSampleConversation(thread_id)
    print(f"Got MultiTurnSample with {len(sample.user_input)} messages")
    
    # Set focused, achievable reference goal
    reference_goal = (
        "Explained what API testing is, described its importance in software development, "
        "and provided clear, accurate information about API testing concepts"
    )
    sample.reference = reference_goal
    
    print(f"Reference goal: {reference_goal}")
    
    # Evaluate with RAGAS
    print("\nRunning RAGAS evaluation...")
    scorer = AgentGoalAccuracyWithReference(llm=langchain_llm_ragas_wrapper)
    score = await scorer.multi_turn_ascore(sample)
    
    print(f"\nResults:")
    print(f"   Score: {score:.3f}")
    print(f"   Threshold: >= 0.7")
    print(f"   Status: {'PASS' if score >= 0.7 else 'FAIL'}")
    
    assert score >= 0.7, f"Goal accuracy score {score:.3f} below threshold"


# File can be run directly with pytest
# Use: pytest tests/test_real_agent_simple.py -v
# This will run all three simplified test functions:
# - test_topic_adherence_simple: Tests QA topic focus (simplified 2-turn conversation)
# - test_tool_accuracy_simple: Tests tool usage for research (simplified 4-message conversation with tool calls)  
# - test_goal_accuracy_simple: Tests goal achievement (simplified 2-turn Q&A)
#
# All tests are simplified for RAGAS compatibility with local Ollama models while maintaining
# comprehensive coverage of agent capabilities within the RAGAS framework.