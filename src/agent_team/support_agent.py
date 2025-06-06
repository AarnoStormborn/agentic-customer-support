""" RAG / Knowledge Base Agent """

import os
from agents import Agent

def init_support_agent(name: str, prompt: str) -> Agent:
    
    agent = Agent(
        name=name,
        instructions=prompt,
        model=os.getenv('OPENAI_MODEL')
    )
    
    return agent
