""" Support Agent - Entry point for customer support """

import os
from typing import List, Optional
from google.adk import Agent
from google.adk.tools import FunctionTool, transfer_to_agent, tool_context
from src.models.litellm_model import LiteLLMModel

from src.logger import logger

def init_support_agent(name: str, prompt: str, sub_agents: Optional[List[Agent]] = None) -> Agent:
    """
    Initialize the Support Agent with optional sub-agents.
    
    Args:
        name: Agent name
        prompt: Agent instructions
    Returns:
        Configured Support Agent
    """
    model = LiteLLMModel(model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'), agent_name=name)
    
    # Wrap transfer_to_agent in FunctionTool so it can be properly converted to OpenAI format
    transfer_tool = FunctionTool(transfer_to_agent)
    
    # Enhance tool description with valid agent names to guide the LLM
    valid_agents = [a.name for a in (sub_agents or [])]
    if valid_agents:
        # Monkeypatch _get_declaration to append allowed values
        # Legacy monkeypatch - keeping it for safety though FunctionTool might be recreated? 
        # Actually FunctionTool is created above.
        
        # Monkeypatch transfer_to_agent wrapper for explicit logging if needed, 
        # BUT transfer-to-agent is handled by ADK.
        # We rely on LiteLLMModel logging which we just added.
        
        # Monkeypatch _get_declaration to append allowed values
        # We need to capture the original method
        orig_get_decl = transfer_tool._get_declaration
        
        def new_get_decl():
            decl = orig_get_decl()
            # Append if not already appended (to avoid duplicates if called multiple times)
            if "Valid agent_name values" not in decl.description:
                # Log that we are enhancing description? No need.
                decl.description += "\n\nValid agent_name values:\n" + "\n".join([f"- '{a}'" for a in valid_agents])
            return decl
            
        transfer_tool._get_declaration = new_get_decl

    agent = Agent(
        name=name,
        instruction=prompt,
        model=model,
        tools=[transfer_tool],
        sub_agents=sub_agents or []
    )
    
    return agent
