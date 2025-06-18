from dotenv import load_dotenv
load_dotenv()

from langchain.agents import initialize_agent, AgentType
from agent.tools_init import tools
from llm.llm_setup import llm

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Change here, back to zero shot, couldnt find a llm to support OpenAI function tool calling
    verbose=True
)
