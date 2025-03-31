""" RAG / Knowledge Base Agent """

import os
from agents import Agent, function_tool


@function_tool
def retriever_tool(query: str):
    
    """
    A function to retrieve information from knowledge base.
    The knowledge base points to a vector database that 
    contains technical manuals
    
    args:
        query: str = The query for which information needs to be retrieved
    returns:
        dict: A dictionary containing relevant information along with source
    """
    
    pass 


def init_rag_agent(name: str, prompt: str) -> Agent:
    
    agent = Agent(
        name=name,
        instructions=prompt,
        tools=[retriever_tool],
        model=os.getenv('OPENAI_MODEL')
    )
    
    return agent