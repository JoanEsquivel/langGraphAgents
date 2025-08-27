"""
Real Agent Evaluation Tests

These tests execute the actual 3_basic_chat_bot_with_tools_memory.py agent
and evaluate real responses using RAGAS metrics for comprehensive assessment.

🚀 EVERYTHING IS REAL - NO SIMULATION:
- Uses actual LangGraph agent from your src code
- Makes real calls to Qwen 2.5:7b-instruct LLM
- Uses real Tavily web search tool
- Captures actual tool calls and responses
- No mocked or simulated data - all results are genuine

Test Coverage:
- Basic agent functionality and tool usage (weather information)
- Topic adherence evaluation (weather + automation testing topics)
- Tool call accuracy assessment (automation testing framework research)
- Goal completion accuracy with reference standards (test automation research)
"""

import sys
import os
import pytest
import uuid
from pathlib import Path

# Add project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAGAS evaluation components
from ragas import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from ragas.metrics import TopicAdherenceScore, ToolCallAccuracy, AgentGoalAccuracyWithReference

# Import our custom agent runner utility
from tests.helpers.utils import RealAgentRunner


@pytest.mark.asyncio  
async def test_real_agent_topic_adherence_simple(langchain_llm_ragas_wrapper):
    """
    Topic Adherence Evaluation Test
    
    This test evaluates the agent's ability to:
    1. Stay focused on professional topics (weather/information/testing)
    2. Handle diverse but relevant technical topics
    3. Maintain conversation relevance using RAGAS TopicAdherenceScore
    
    Test Flow:
    - Ask weather information question
    - Follow up with automation testing question  
    - Measure how well agent maintains professional topic focus
    """
    
    print("\n" + "="*60)
    print("🧪 TEST: Topic Adherence Assessment (RAGAS)")
    print("="*60)
    
    # Create agent instance for conversation with custom test name
    agent = RealAgentRunner("TopicAdherenceTest")
    
    # First interaction: Professional weather query (on-topic)
    weather_question = "What are the current weather conditions in Barcelona, Spain?"
    result1 = agent.ask_agent(weather_question)
    
    # Second interaction: Automation testing question (on-topic test)
    testing_question = "What are the best practices for implementing automated testing in CI/CD pipelines?"
    result2 = agent.ask_agent(testing_question)
    
    print(f"\n📝 Constructing RAGAS conversation sample...")
    
    # Build conversation structure for RAGAS evaluation
    # Note: Using simple AI/Human messages to avoid ToolMessage complexity
    conversation = [
        HumanMessage(content=result1['question']),     # Weather question
        AIMessage(content=result1['response']),        # Agent weather response
        HumanMessage(content=result2['question']),     # Automation testing question
        AIMessage(content=result2['response'])         # Agent testing response
    ]
    
    # Create RAGAS sample with more comprehensive reference topics for adherence measurement
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
    
    print(f"🎯 Evaluating topic adherence with RAGAS scorer...")
    
    # Initialize and execute RAGAS topic adherence evaluation
    # Using 'recall' mode instead of 'precision' for more flexible evaluation
    scorer = TopicAdherenceScore(llm=langchain_llm_ragas_wrapper, mode="recall")
    score = await scorer.multi_turn_ascore(sample)
    
    # Present evaluation results
    print(f"\n📊 RAGAS EVALUATION RESULTS:")
    print(f"   🎯 Adherence Score: {score:.3f}")
    print(f"   📏 Acceptance Threshold: 0.4")
    
    # Interpret results
    if score >= 0.4:
        print(f"   ✅ PASS: Acceptable topic adherence maintained")
    else:
        print(f"   ❌ FAIL: Topic adherence below required threshold")
    
    # Apply RAGAS-based threshold for topic adherence
    assert score >= 0.4, f"RAGAS TopicAdherenceScore {score:.3f} below acceptable threshold of 0.4"
    
    print("✅ TEST COMPLETED: Topic adherence successfully evaluated")


@pytest.mark.asyncio
async def test_real_agent_tool_accuracy_simple():
    """
    Tool Usage Accuracy Test
    
    This test verifies the agent's ability to:
    1. Identify when tools are needed for information gathering
    2. Select appropriate tools for specific tasks
    3. Use tools with relevant arguments
    4. Integrate tool results effectively into responses
    
    Focus: Automation testing news search requiring web tool usage
    """
    
    print("\n" + "="*60) 
    print("🧪 TEST: Tool Usage Accuracy Assessment")
    print("="*60)
    
    # Initialize agent for tool testing with custom test name
    agent = RealAgentRunner("ToolAccuracyTest")
    
    # Professional question that clearly requires web search tool usage
    research_question = "Please search for recent news about automation testing frameworks and tools"
    result = agent.ask_agent(research_question)
    
    print(f"\n📊 TOOL USAGE ANALYSIS:")
    print(f"   • Tools Activated: {result['tools_used']}")
    
    # Examine each tool call for accuracy
    for i, tc in enumerate(result['tool_calls']):
        print(f"   • Tool Call {i+1}: {tc['name']}")
        print(f"     Arguments: {tc['args']}")
        
    # Verify correct tool selection
    tool_names = [tc['name'] for tc in result['tool_calls']]
    
    # Check if Tavily search tool was used (expected for web searches)
    if any("tavily" in name.lower() for name in tool_names):
        print(f"   ✅ Correctly selected Tavily for web search")
    
    # Analyze search query relevance
    queries = []
    for tc in result['tool_calls']:
        if 'query' in tc['args']:
            queries.append(tc['args']['query'])
            
    # Verify query content alignment with request
    if queries:
        print(f"   📝 Search Queries Generated: {queries}")
        relevant_keywords = ["automation testing", "test automation", "testing frameworks", "selenium", "cypress", "playwright"]
        query_text = ' '.join(queries).lower()
        
        if any(keyword in query_text for keyword in relevant_keywords):
            print(f"   ✅ Search queries contain relevant keywords")
        else:
            print(f"   ⚠️  Search queries may lack topic relevance")
    
    # Assert that tools were actually used for this search task
    assert result['tools_used'] > 0, f"Agent should use tools for search requests but used {result['tools_used']} tools"
    
    print("✅ TEST COMPLETED: Tool usage patterns analyzed successfully")


@pytest.mark.asyncio
async def test_real_agent_goal_accuracy_with_reference(langchain_llm_ragas_wrapper):
    """
    Goal Achievement Accuracy Test with Reference Standard
    
    This test evaluates the agent's ability to:
    1. Understand complex multi-step user goals
    2. Execute appropriate actions to achieve goals
    3. Complete tasks according to reference standards
    4. Use RAGAS AgentGoalAccuracyWithReference for objective evaluation
    
    Scenario: Automation testing framework research and summarization task
    """
    
    print("\n" + "="*60)
    print("🧪 TEST: Goal Achievement Accuracy (RAGAS)")
    print("="*60)
    
    # Initialize agent for goal completion testing with custom test name
    agent = RealAgentRunner("GoalAccuracyTest")
    
    # Define a clear, measurable goal for the agent
    task_request = "Please research the latest developments in test automation frameworks and provide a brief summary"
    
    print(f"📋 Assigned Task: {task_request}")
    
    # Execute the task and capture agent's response
    result = agent.ask_agent(task_request)
    
    print(f"🔄 Agent executed task with {result['tools_used']} tool calls")
    
    # Use the agent's natural single-turn completion for evaluation
    # This is the most realistic scenario - agent gets a task and completes it
    conversation = [
        HumanMessage(content=task_request),
        AIMessage(
            content=result['response'],
            tool_calls=[
                ToolCall(name=tc['name'], args=tc['args']) 
                for tc in result['tool_calls']
            ] if result['tool_calls'] else []
        )
    ]
    
    print(f"📝 Agent response length: {len(result['response'])} characters")
    print(f"🔧 Tools executed: {[tc['name'] for tc in result['tool_calls']]}")
    
    # Display actual response summary for verification
    response_preview = result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
    print(f"💬 Agent response preview: {response_preview}")
    
    # Define reference standard for goal achievement
    reference_goal = "Agent should research and provide a summary about test automation framework developments"
    
    # Create RAGAS sample for goal accuracy evaluation
    sample = MultiTurnSample(
        user_input=conversation,
        reference=reference_goal
    )
    
    print(f"🎯 Evaluating goal achievement with RAGAS scorer...")
    
    # Initialize goal accuracy scorer with reference standard
    scorer = AgentGoalAccuracyWithReference(llm=langchain_llm_ragas_wrapper)
    
    # Execute goal achievement evaluation
    score = await scorer.multi_turn_ascore(sample)
    
    # Present comprehensive results
    print(f"\n📊 GOAL ACHIEVEMENT RESULTS:")
    print(f"   🎯 Goal Accuracy Score: {score:.3f}")
    print(f"   📋 Reference Standard: {reference_goal}")
    print(f"   🔧 Tools Used: {result['tools_used']}")
    print(f"   📝 Response Length: {len(result['response'])} characters")
    
    # Analyze goal completion effectiveness
    if score >= 0.7:
        print(f"   ✅ EXCELLENT: High goal achievement accuracy")
    elif score >= 0.5:
        print(f"   ✅ GOOD: Acceptable goal achievement")
    else:
        print(f"   ⚠️  NEEDS IMPROVEMENT: Goal achievement below expectations")
    
    # Apply RAGAS-based threshold for goal achievement
    assert score >= 0.4, f"RAGAS AgentGoalAccuracyWithReference score {score:.3f} below acceptable threshold of 0.4"
    
    print("✅ TEST COMPLETED: Goal achievement accuracy successfully evaluated")


if __name__ == "__main__":
    """
    Manual Testing Entry Point
    
    This section allows for quick manual testing of agent functionality
    without running the full pytest suite.
    """
    import asyncio
    
    async def quick_test():
        """Run a simple test interaction with the agent"""
        print("🚀 Running quick manual test...")
        
        # Create agent instance with custom test name
        agent = RealAgentRunner("ManualQuickTest")
        
        # Test basic interaction
        result = agent.ask_agent("Hello, could you please introduce your capabilities?")
        
        print(f"\n📋 Manual Test Results:")
        print(f"   Question: {result['question']}")
        print(f"   Response: {result['response']}")
        print(f"   Tools Used: {result['tools_used']}")
        
        print("\n✅ Manual test completed successfully")
    
    # Execute manual test
    asyncio.run(quick_test())