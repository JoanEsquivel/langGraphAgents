"""
Agent Goal Accuracy Test

Simple test based on RAGAS examples to evaluate if the agent achieves the user's goal
of getting comprehensive information about a topic through research.
"""

import sys
import os
import pytest
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAGAS components for agent evaluation
from ragas import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from ragas.metrics import AgentGoalAccuracyWithReference


@pytest.mark.asyncio
async def test_agent_goal_accuracy_research_task(langchain_llm_ragas_wrapper):
    """
    Test agent goal accuracy: comprehensive research about electric vehicles.
    Based on RAGAS example pattern.
    """
    
    # Create a conversation sample similar to RAGAS example
    # User wants comprehensive information about electric vehicles
    conversation = [
        HumanMessage(content="I need to research electric vehicles for a school project. Can you help me find information about their benefits and current market?"),
        AIMessage(content="I'd be happy to help you with your electric vehicle research! Let me search for information about their benefits.", tool_calls=[
            ToolCall(name="tavily_search_results_json", args={"query": "electric vehicles benefits environmental economic"})
        ]),
        ToolMessage(content="Electric vehicles offer several benefits: zero direct emissions, lower operating costs, reduced maintenance needs, and government incentives. They help reduce air pollution and dependence on fossil fuels."),
        AIMessage(content="Great! I found information about EV benefits including environmental advantages and cost savings. Now let me search for current market information."),
        HumanMessage(content="That's helpful. What about the current market situation?"),
        AIMessage(content="Let me find current market data for electric vehicles.", tool_calls=[
            ToolCall(name="tavily_search_results_json", args={"query": "electric vehicle market 2024 sales statistics"})
        ]),
        ToolMessage(content="The 2024 EV market shows strong growth with over 10 million EVs sold globally. Major manufacturers like Tesla, BYD, and traditional automakers are expanding their EV lineups. Market share reached 18% of new vehicle sales."),
        AIMessage(content="Excellent! Here's what I found for your project:\n\n**Benefits of Electric Vehicles:**\n- Zero direct emissions (better for environment)\n- Lower operating costs\n- Reduced maintenance needs\n- Government incentives available\n\n**Current Market (2024):**\n- Over 10 million EVs sold globally\n- 18% market share of new vehicle sales\n- Major growth from Tesla, BYD, and traditional manufacturers\n\nThis should give you a solid foundation for your school project!"),
        HumanMessage(content="Perfect, thank you!")
    ]
    
    # Create the sample with reference goal (what the user wanted to achieve)
    sample = MultiTurnSample(
        user_input=conversation,
        reference="Comprehensive research information provided about electric vehicles including benefits and current market data for school project"
    )
    
    # Initialize the metric
    agent_goal_accuracy = AgentGoalAccuracyWithReference(llm=langchain_llm_ragas_wrapper)
    
    # Calculate score
    score = await agent_goal_accuracy.multi_turn_ascore(sample)
    
    print(f"\n🎯 Agent Goal Accuracy Score: {score:.3f}")
    print("✅ Expected: Agent should provide comprehensive EV information")
    print("✅ Expected: Agent should address both benefits and market data")
    print("✅ Expected: Information should be suitable for school project")
    
    # Assert score meets threshold - should be high since goal is achieved
    assert score >= 0.7, f"Agent goal accuracy score {score} is below threshold 0.7"
    
    print(f"✅ PASSED: Agent goal accuracy score {score:.3f} meets threshold")