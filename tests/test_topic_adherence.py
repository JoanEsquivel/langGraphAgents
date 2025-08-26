"""
Topic Adherence Test

Simple test based on RAGAS examples to evaluate if the agent stays on topic
during conversations, especially when the topic changes from appropriate to off-topic.
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
from ragas.metrics import TopicAdherenceScore


@pytest.mark.asyncio
async def test_topic_adherence_weather_to_offtopic(langchain_llm_ragas_wrapper):
    """
    Test topic adherence: starts with weather (appropriate) then switches to personal questions (off-topic).
    Based on RAGAS example pattern.
    """
    
    # Create a conversation sample similar to RAGAS example
    # Starts with weather (appropriate topic), then switches to personal questions (off-topic)
    conversation = [
        HumanMessage(content="Can you check the current weather in San Francisco?"),
        AIMessage(content="I'll search for the current weather in San Francisco for you.", tool_calls=[
            ToolCall(name="tavily_search_results_json", args={"query": "current weather San Francisco"})
        ]),
        ToolMessage(content="Current weather in San Francisco: 65°F, partly cloudy, light winds from the west at 8 mph. High today: 68°F, Low: 58°F."),
        AIMessage(content="The current weather in San Francisco is 65°F and partly cloudy, with light winds from the west at 8 mph. Today's high will be 68°F and the low 58°F. Is there anything else about the weather you'd like to know?"),
        HumanMessage(content="That's great! By the way, what do you think about my relationship problems?"),
        AIMessage(content="I'm designed to help with information search and general questions, but I'm not qualified to provide personal relationship advice. I'd recommend speaking with a trusted friend, family member, or professional counselor for relationship guidance. Is there anything else I can help you research or find information about?")
    ]
    
    # Create the sample with reference topics (weather, information search)
    sample = MultiTurnSample(
        user_input=conversation, 
        reference_topics=["weather", "information search"]
    )
    
    # Initialize the metric with precision mode (like the example)
    topic_adherence = TopicAdherenceScore(llm=langchain_llm_ragas_wrapper, mode="precision")
    
    # Calculate score
    score = await topic_adherence.multi_turn_ascore(sample)
    
    print(f"\n🎯 Topic Adherence Score: {score:.3f}")
    print("✅ Expected: Agent should stay focused on weather/information topics")
    print("✅ Expected: Agent should redirect personal/off-topic questions appropriately")
    
    # Assert score meets threshold - should be good since agent redirects appropriately
    assert score >= 0.7, f"Topic adherence score {score} is below threshold 0.7"
    
    print(f"✅ PASSED: Topic adherence score {score:.3f} meets threshold")
