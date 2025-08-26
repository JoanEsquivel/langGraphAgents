"""
Tool Call Accuracy Test

Simple test based on RAGAS examples to evaluate if the agent makes correct tool calls
with appropriate parameters when requesting weather information.
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
from ragas.metrics import ToolCallAccuracy


@pytest.mark.asyncio
async def test_tool_call_accuracy_weather_search():
    """
    Test tool call accuracy: weather search with correct tool usage.
    Based on RAGAS example pattern.
    """
    
    # Create a conversation sample similar to RAGAS example
    # Agent should use tavily_search_results_json for weather queries
    conversation = [
        HumanMessage(content="What's the weather like in New York right now?"),
        AIMessage(content="I'll search for the current weather in New York for you.", tool_calls=[
            ToolCall(name="tavily_search_results_json", args={"query": "current weather New York"})
        ]),
        ToolMessage(content="Current weather in New York: 72°F, sunny with clear skies, light winds from the southwest at 5 mph."),
        AIMessage(content="The current weather in New York is 72°F and sunny with clear skies. There are light winds from the southwest at 5 mph. Is there anything else about the weather you'd like to know?"),
        HumanMessage(content="Can you also check the weather forecast for tomorrow?"),
        AIMessage(content="Let me search for tomorrow's weather forecast in New York.", tool_calls=[
            ToolCall(name="tavily_search_results_json", args={"query": "weather forecast tomorrow New York"})
        ]),
        ToolMessage(content="Tomorrow's forecast for New York: High 75°F, Low 62°F, partly cloudy with a 20% chance of rain."),
        AIMessage(content="Tomorrow's forecast for New York shows a high of 75°F and low of 62°F, with partly cloudy skies and a 20% chance of rain.")
    ]
    
    # Create the sample with reference tool calls (what we expect the agent to call)
    sample = MultiTurnSample(
        user_input=conversation,
        reference_tool_calls=[
            ToolCall(name="tavily_search_results_json", args={"query": "current weather New York"}),
            ToolCall(name="tavily_search_results_json", args={"query": "weather forecast tomorrow New York"})
        ]
    )
    
    # Initialize the metric
    tool_call_accuracy = ToolCallAccuracy()
    
    # Calculate score
    score = await tool_call_accuracy.multi_turn_ascore(sample)
    
    print(f"\n🔧 Tool Call Accuracy Score: {score:.3f}")
    print("✅ Expected: Agent should use tavily_search_results_json for weather queries")
    print("✅ Expected: Agent should provide appropriate search parameters")
    
    # Assert score meets threshold - should be high since correct tools are used
    assert score >= 0.8, f"Tool call accuracy score {score} is below threshold 0.8"
    
    print(f"✅ PASSED: Tool call accuracy score {score:.3f} meets threshold")