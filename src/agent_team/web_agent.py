""" Web Search Agent """

import os
from google.adk import Agent
from google.adk.tools import FunctionTool
from src.models.litellm_model import LiteLLMModel
from duckduckgo_search import DDGS

def web_search(query: str) -> str:
    """
    Search the web for the given query.
    
    Args:
        query: The search query.
        
    Returns:
        String containing search results.
    """
    try:
        results = DDGS().text(query, max_results=5)
        return str(results)
    except Exception as e:
        return f"Error searching web: {e}"

def init_web_agent(name, prompt) -> Agent:
    model = LiteLLMModel(model=os.getenv("OPENAI_MODEL"))
    
    agent = Agent(
        name=name,
        tools=[FunctionTool(web_search)],
        instruction=prompt,
        model=model
    )
    
    return agent
