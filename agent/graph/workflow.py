"""
Main workflow definition for the Stock Insight Agent.
This file compiles the LangGraph workflow with nodes and edges.
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage

# Import nodes and edges
from agent.graph.nodes.tool_caller import call_tool, AgentState
from agent.graph.edges.decision_router import should_continue

def create_workflow():
    """
    Creates and compiles the LangGraph workflow for the Stock Insight Agent.
    
    Returns:
        Compiled LangGraph workflow
    """
    # Create the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("call_tool", call_tool)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "call_tool",
        should_continue,
        {
            "continue": "call_tool",
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("call_tool")
    
    # Compile and return
    return workflow.compile()

# Create the compiled workflow
app = create_workflow()
