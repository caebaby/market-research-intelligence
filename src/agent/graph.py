"""
Market Research Intelligence Platform
Production-grade customer psychology research
"""

from typing import Annotated, Dict, Any
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph import StateGraph, END
from langgraph.graph.message import add_messages
from langsmith import traceable
import os

class ResearchState(TypedDict):
    """State for market research workflow"""
    messages: Annotated[list[BaseMessage], add_messages]
    business_context: str
    research_insights: str
    quality_score: int
    session_id: str

# Production-grade research prompt
PSYCHOLOGY_RESEARCH_PROMPT = """
MARKET RESEARCH INTELLIGENCE - SESSION ISOLATION PROTOCOL

BUSINESS CONTEXT FOR ANALYSIS:
{business_context}

RESEARCH DIRECTIVE:
Conduct comprehensive customer psychology research with Eugene Schwartz-level depth and business actionability.

ANALYSIS FRAMEWORK:

1. PSYCHOLOGICAL PROFILING
   - Core motivational drivers and psychological triggers
   - Hidden pain points and unconscious desires
   - Belief system analysis and mental models
   - Decision-making patterns and cognitive biases

2. VOICE OF CUSTOMER INTELLIGENCE
   - Authentic language patterns and terminology
   - Emotional expression patterns
   - Pain articulation phrases and frustration language
   - Aspiration and desire language patterns

3. BUYER PSYCHOLOGY FRAMEWORKS
   - Eugene Schwartz awareness levels (unaware → most aware)
   - Decision triggers and conversion psychology
   - Objection patterns and resistance points
   - Trust and authority preference indicators

4. STRATEGIC POSITIONING INSIGHTS
   - Breakthrough positioning opportunities
   - Messaging angles by awareness level
   - Conversion psychology applications
   - Competitive differentiation psychology

5. CAMPAIGN-READY INTELLIGENCE
   - Specific copy hooks and headlines
   - Emotional trigger sequences
   - Trust-building messaging strategy
   - Conversion path psychology

QUALITY STANDARDS:
- Provide insights that make clients say "how did you know that?"
- Include 3-5 authentic customer voice examples
- Ensure all insights are immediately actionable
- Focus on psychological depth beyond surface demographics

DELIVERABLE FORMAT:
Structure response with clear sections, authentic quotes, and specific recommendations.
"""

@traceable(name="psychology_research_node")
def research_node(state: ResearchState) -> Dict[str, Any]:
    """
    Main psychology research node
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Format research prompt
    formatted_prompt = PSYCHOLOGY_RESEARCH_PROMPT.format(
        business_context=state["business_context"]
    )
    
    # Execute research
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    
    # Quality scoring based on response length and depth
    quality_score = min(95, max(70, len(response.content) // 50))
    
    return {
        "research_insights": response.content,
        "quality_score": quality_score,
        "messages": [response],
        "session_id": f"research_{hash(state['business_context']) % 10000}"
    }

# Create production research workflow
def create_research_workflow():
    """
    Production workflow for market research
    """
    workflow = StateGraph(ResearchState)
    
    # Add research node
    workflow.add_node("research", research_node)
    
    # Define flow
    workflow.set_entry_point("research")
    workflow.add_edge("research", END)
    
    return workflow.compile()

# Export the compiled graph for LangGraph Platform
graph = create_research_workflow()
