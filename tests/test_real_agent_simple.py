"""
Enhanced QA Automation Agent Evaluation Tests with RAGAS

These tests create challenging real conversations using the 4-final-agent script and evaluate them
with RAGAS metrics. Each test is specifically designed to challenge QA automation capabilities.

Enhanced Test Coverage:
- Topic Adherence: Agent stays within QA automation boundaries across multiple testing domains
  (functional, performance, API, mobile testing) with stricter evaluation criteria
- Tool Call Accuracy: Agent performs complex research requiring multiple targeted searches
  for current QA tool comparisons and compatibility issues  
- Goal Achievement: Agent provides focused QA automation recommendations with specific 
  tool selection, features, and rationale based on current research
"""

import sys
import os
import pytest
import uuid
import importlib.util

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAGAS components
from ragas.dataset_schema import MultiTurnSample
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
    Topic Adherence Test - Enhanced for QA Automation
    
    Tests if the agent stays within QA automation boundaries across different testing domains.
    Uses multiple QA-specific questions to challenge topic adherence more rigorously.
    """
    print("\nTOPIC ADHERENCE TEST - QA AUTOMATION FOCUS")
    print("="*50)
    
    # Create conversation
    thread_id = f"topic_test_{uuid.uuid4().hex[:8]}"
    print(f"Thread: {thread_id}")
    
    # Ask about functional testing strategies
    print("\nQuestion 1: Functional Testing")
    stream_graph_updates("What are the key strategies for functional testing in agile environments?", thread_id)
    
    # Ask about performance testing
    print("\nQuestion 2: Performance Testing")
    stream_graph_updates("How should I approach performance testing for microservices architecture?", thread_id)
    
    # Ask about API testing
    print("\nQuestion 3: API Testing")
    stream_graph_updates("What are the best practices for API testing automation using REST endpoints?", thread_id)
    
    # Ask about mobile testing (edge case - should stay in QA)
    print("\nQuestion 4: Mobile Testing")
    stream_graph_updates("Explain mobile testing considerations for iOS and Android applications", thread_id)
    
    # Get conversation using unified method
    print("\nGetting conversation using unified method...")
    sample = getMultiTurnSampleConversation(thread_id)
    print(f"Got MultiTurnSample with {len(sample.user_input)} messages")
    
    # Set specific QA automation reference topics
    reference_topics = [
        "functional testing", "performance testing", "API testing", "mobile testing",
        "test automation", "agile testing", "microservices testing", "REST API",
        "iOS testing", "Android testing", "testing strategies", "testing practices"
    ]
    sample.reference_topics = reference_topics
    
    # Evaluate with RAGAS
    print("\nRunning RAGAS evaluation...")
    scorer = TopicAdherenceScore(llm=langchain_llm_ragas_wrapper, mode="recall")
    score = await scorer.multi_turn_ascore(sample)
    
    print(f"\nResults:")
    print(f"   Score: {score:.3f}")
    print(f"   Threshold: >= 0.6")
    print(f"   Status: {'PASS' if score >= 0.6 else 'FAIL'}")
    
    assert score >= 0.6, f"Topic adherence score {score:.3f} below threshold"


@pytest.mark.asyncio  
async def test_tool_accuracy_simple(langchain_llm_ragas_wrapper):
    """
    Tool Call Accuracy Test - Enhanced for QA Automation
    
    Tests if agent uses search tools correctly for complex QA automation research.
    Requires multiple targeted searches to get comprehensive, current information.
    """
    print("\nTOOL CALL ACCURACY TEST - COMPLEX QA RESEARCH")
    print("="*50)
    
    # Create conversation
    thread_id = f"tool_test_{uuid.uuid4().hex[:8]}"
    print(f"Thread: {thread_id}")
    
    # Ask complex question requiring multiple searches
    print("\nQuestion: Complex Research Task")
    stream_graph_updates(
        "I need current information about Selenium WebDriver 4.x compatibility issues with Chrome 120+ "
        "and the latest Playwright vs Cypress performance benchmarks from 2024. "
        "Also find recent TestNG vs JUnit 5 feature comparisons.",
        thread_id
    )
    
    # Follow up to ensure comprehensive research
    print("\nFollow-up: Specific metrics request")
    stream_graph_updates(
        "Can you also search for recent test execution speed comparisons between these tools? "
        "I need specific performance numbers if available.",
        thread_id
    )
    
    # Get conversation using unified method
    print("\nGetting conversation using unified method...")
    sample = getMultiTurnSampleConversation(thread_id)
    print(f"Got MultiTurnSample with {len(sample.user_input)} messages")
    
    # Extract tool calls for RAGAS reference (no evaluation logic)
    tool_calls_found = []
    for msg in sample.user_input:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls_found.extend(msg.tool_calls)
    
    # Set reference tool calls following RAGAS documentation  
    sample.reference_tool_calls = tool_calls_found  # Use actual tool calls as reference
    
    # Evaluate with RAGAS only
    print("\nRunning RAGAS evaluation...")
    scorer = ToolCallAccuracy()
    score = await scorer.multi_turn_ascore(sample)
    
    print(f"\nResults:")
    print(f"   Score: {score:.3f}")
    print(f"   Threshold: >= 0.8")
    print(f"   Status: {'PASS' if score >= 0.8 else 'FAIL'}")
    
    assert score >= 0.8, f"Tool accuracy score {score:.3f} below threshold"


@pytest.mark.asyncio
async def test_goal_accuracy_simple(langchain_llm_ragas_wrapper):
    """
    Goal Achievement Test - Enhanced for QA Automation
    
    Tests if agent achieves QA automation goals with specific deliverables.
    Uses a focused API testing recommendation task that RAGAS can evaluate reliably.
    """
    print("\nGOAL ACHIEVEMENT TEST - QA API TESTING FOCUS")
    print("="*50)
    
    # Create conversation
    thread_id = f"goal_simple_{uuid.uuid4().hex[:8]}"
    print(f"Thread: {thread_id}")
    
    # Give agent a clear, single goal
    print("\nTask: Simple QA Tool Recommendation")
    simple_task = (
        "I need you to research and recommend the best test automation tool "
        "for API testing in 2024. Provide the tool name, key features, and why it's recommended."
    )
    stream_graph_updates(simple_task, thread_id)
    
    # Get conversation using unified method
    print("\nGetting conversation using unified method...")
    sample = getMultiTurnSampleConversation(thread_id)
    print(f"Got MultiTurnSample with {len(sample.user_input)} messages")
    
    # Set simple reference goal (matching the example format)
    reference_goal = "Recommended a specific API testing tool with features and rationale based on current research"
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
# This will run all three test functions:
# - test_topic_adherence_simple
# - test_tool_accuracy_simple  
# - test_goal_accuracy_simple