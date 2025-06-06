""" Web Search Agent """

import os
from agents import Agent, WebSearchTool


def init_web_agent(name, prompt) -> Agent:
    agent = Agent(
        name=name,
        tools=[WebSearchTool()],
        instructions=prompt,
        model=os.getenv("OPENAI_MODEL")
    )
    
    return agent
