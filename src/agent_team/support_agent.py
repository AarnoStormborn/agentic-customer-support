""" Support Agent - Entry point for customer support """

import os
from typing import List, Optional
from google.adk import Agent
from src.models.litellm_model import LiteLLMModel

def init_support_agent(name: str, prompt: str, sub_agents: Optional[List[Agent]] = None) -> Agent:
    """
    Initialize the Support Agent with optional sub-agents.
    
    Args:
        name: Agent name
        prompt: Agent instructions
        sub_agents: List of sub-agents (RAG, SQL, Web) for delegation
        
    Returns:
        Configured Support Agent
    """
    model = LiteLLMModel(model=os.getenv('OPENAI_MODEL'))
    
    agent = Agent(
        name=name,
        instruction=prompt,
        model=model,
        sub_agents=sub_agents or []
    )
    
    return agent

