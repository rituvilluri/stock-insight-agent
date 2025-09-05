"""
Decision routing edge for the Stock Insight Agent.
This edge determines whether to continue processing or end the workflow.
"""

from langchain_core.messages import AIMessage
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    next: Annotated[str, "The next action to take"]

def should_continue(state: AgentState) -> str:
    """
    Determines whether to continue processing or end the workflow.
    
    Args:
        state: The current agent state containing messages
        
    Returns:
        str: "end" if the last message is from AI, "continue" otherwise
    """
    # If the last message is from the AI, we're done
    if isinstance(state["messages"][-1], AIMessage):
        return "end"
    return "continue"
