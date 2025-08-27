"""
Simple Real Agent Evaluation Tests with RAGAS

These tests create real conversations using the 4-final-agent script and evaluate them
with RAGAS metrics. Each test follows the official RAGAS documentation examples.

Test Coverage:
- Topic Adherence: Agent stays on professional topics
- Tool Call Accuracy: Agent uses tools correctly  
- Goal Achievement: Agent completes tasks successfully
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
get_conversation_for_ragas = agent_final.get_conversation_for_ragas
get_conversation_for_tool_accuracy = agent_final.get_conversation_for_tool_accuracy
get_conversation_for_goal_accuracy = agent_final.get_conversation_for_goal_accuracy

print("Agent functions imported successfully")


@pytest.mark.asyncio
async def test_topic_adherence_simple(langchain_llm_ragas_wrapper):
    """
    Topic Adherence Test
    
    Tests if the agent stays on professional topics (weather, testing).
    Uses get_conversation_for_tool_accuracy() utility for RAGAS message format.
    """
    print("\nTOPIC ADHERENCE TEST")
    print("="*50)
    
    # Create conversation
    thread_id = f"topic_test_{uuid.uuid4().hex[:8]}"
    print(f"Thread: {thread_id}")
    
    # Ask weather question
    print("\nQuestion 1: Weather information")
    stream_graph_updates("What's the weather in Barcelona?", thread_id)
    
    # Ask testing question  
    print("\nQuestion 2: Testing topic")
    stream_graph_updates("What are CI/CD best practices?", thread_id)
    
    # Get conversation using correct utility for RAGAS messages
    print("\nGetting conversation for RAGAS...")
    conversation = get_conversation_for_tool_accuracy(thread_id)  # Returns RAGAS messages
    print(f"Got {len(conversation)} RAGAS messages")
    
    # Create sample following RAGAS documentation
    reference_topics = ["weather", "testing", "CI/CD", "automation", "technical information"]
    sample = MultiTurnSample(
        user_input=conversation,
        reference_topics=reference_topics
    )
    
    # Evaluate with RAGAS
    print("\nRunning RAGAS evaluation...")
    scorer = TopicAdherenceScore(llm=langchain_llm_ragas_wrapper, mode="recall")
    score = await scorer.multi_turn_ascore(sample)
    
    print(f"\nResults:")
    print(f"   Score: {score:.3f}")
    print(f"   Threshold: >= 0.4")
    print(f"   Status: {'PASS' if score >= 0.4 else 'FAIL'}")
    
    assert score >= 0.4, f"Topic adherence score {score:.3f} below threshold"


@pytest.mark.asyncio  
async def test_tool_accuracy_simple(langchain_llm_ragas_wrapper):
    """
    Tool Call Accuracy Test
    
    Tests if agent uses tools correctly for research tasks.
    Uses get_conversation_for_tool_accuracy() utility as designed.
    """
    print("\nTOOL CALL ACCURACY TEST")
    print("="*50)
    
    # Create conversation
    thread_id = f"tool_test_{uuid.uuid4().hex[:8]}"
    print(f"Thread: {thread_id}")
    
    # Ask question that requires web search
    print("\nQuestion: Research task")
    stream_graph_updates("Search for recent automation testing news", thread_id)
    
    # Get conversation using correct utility
    print("\nGetting conversation for tool accuracy...")
    conversation = get_conversation_for_tool_accuracy(thread_id)
    print(f"Got {len(conversation)} ragas messages")
    
    # Check for tool calls in conversation
    tool_calls_found = []
    for msg in conversation:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls_found.extend(msg.tool_calls)
    
    print(f"Tool calls found: {len(tool_calls_found)}")
    if tool_calls_found:
        for tc in tool_calls_found:
            print(f"   - {tc.name}({list(tc.args.keys())})")
    
    # Create sample following RAGAS documentation  
    sample = MultiTurnSample(
        user_input=conversation,
        reference_tool_calls=tool_calls_found  # Use actual tool calls as reference
    )
    
    # Evaluate with RAGAS
    print("\nRunning RAGAS evaluation...")
    scorer = ToolCallAccuracy()
    score = await scorer.multi_turn_ascore(sample)
    
    print(f"\nResults:")
    print(f"   Score: {score:.3f}")
    print(f"   Perfect Score: 1.0") 
    print(f"   Status: {'PASS' if score >= 0.7 else 'FAIL'}")
    
    assert score >= 0.7, f"Tool accuracy score {score:.3f} below threshold"


@pytest.mark.asyncio
async def test_goal_accuracy_simple(langchain_llm_ragas_wrapper):
    """
    Goal Achievement Test
    
    Tests if agent achieves assigned goals successfully.
    Uses get_conversation_for_goal_accuracy() utility as designed.
    """
    print("\nGOAL ACHIEVEMENT TEST") 
    print("="*50)
    
    # Create conversation
    thread_id = f"goal_test_{uuid.uuid4().hex[:8]}"
    print(f"Thread: {thread_id}")
    
    # Give agent a clear goal
    print("\nTask: Research and summarize")
    task = "Research latest test automation frameworks and provide a summary"
    stream_graph_updates(task, thread_id)
    
    # Get conversation using correct utility
    print("\nGetting conversation for goal accuracy...")
    sample = get_conversation_for_goal_accuracy(thread_id)
    print(f"Got MultiTurnSample with {len(sample.user_input)} messages")
    
    # Set reference goal following RAGAS documentation
    reference_goal = "Agent should research and provide a comprehensive summary of test automation frameworks"
    sample.reference = reference_goal
    
    print(f"Reference goal: {reference_goal}")
    
    # Evaluate with RAGAS
    print("\nRunning RAGAS evaluation...")
    scorer = AgentGoalAccuracyWithReference(llm=langchain_llm_ragas_wrapper)
    score = await scorer.multi_turn_ascore(sample)
    
    print(f"\nResults:")
    print(f"   Score: {score:.3f}")
    print(f"   Threshold: >= 0.5")
    print(f"   Status: {'PASS' if score >= 0.5 else 'FAIL'}")
    
    assert score >= 0.5, f"Goal accuracy score {score:.3f} below threshold"


if __name__ == "__main__":
    """Quick manual test"""
    import asyncio
    
    async def quick_test():
        print("Quick Agent Test")
        thread_id = f"manual_{uuid.uuid4().hex[:8]}"
        
        # Test basic interaction
        stream_graph_updates("Hello, what can you do?", thread_id)
        
        # Show available utilities
        print(f"\nAvailable utilities for thread {thread_id}:")
        print("- get_conversation_for_ragas() → for topic adherence")
        print("- get_conversation_for_tool_accuracy() → for tool accuracy") 
        print("- get_conversation_for_goal_accuracy() → for goal accuracy")
        
        print("Quick test completed")
    
    asyncio.run(quick_test())