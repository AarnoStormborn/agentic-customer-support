""" Web Search Agent """

import os
from google.adk import Agent
from google.adk.tools import FunctionTool, transfer_to_agent
from src.models.litellm_model import LiteLLMModel
from duckduckgo_search import DDGS
from src.logger import logger

def web_search(query: str) -> str:
    """
    Search the web for the given query.
    
    Args:
        query: The search query.
        
    Returns:
        String containing search results.
    """
    try:
        logger.info(f"Searching web for: {query}")
        results = DDGS().text(query, max_results=5)
        return str(results)
    except Exception as e:
        return f"Error searching web: {e}"

def init_web_agent(name: str, prompt: str, description: str = "") -> Agent:
    model = LiteLLMModel(model=os.getenv("OPENAI_MODEL"), agent_name=name)
    
    agent = Agent(
        name=name,
        description=description or "Web search agent that finds information from the internet",
        tools=[FunctionTool(web_search), FunctionTool(transfer_to_agent)],
        instruction=prompt,
        model=model
    )
    
    return agent

