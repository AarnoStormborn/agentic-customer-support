import os

from src.agent_team import (
    init_rag_agent,
    init_sql_agent,
    init_web_agent,
    init_support_agent
)
from src.guardrails import (
    init_supervisor_guardrail
)
from src.utils import read_config

from agents import Runner
from agents.exceptions import InputGuardrailTripwireTriggered

from src.logger import logger
from src.exception import CustomException


class SupportOrchestrator:
    
    def __init__(self):
        
        self.config = read_config(os.getenv("AGENTS_CONFIG_FILE_PATH"))
        self.support_agent = init_support_agent(**self.config.agents.support_agent)
        self.rag_agent = init_rag_agent(**self.config.agents.rag_agent)
        self.sql_agent = init_sql_agent(**self.config.agents.sql_agent)
        self.web_agent = init_web_agent(**self.config.agents.web_agent)
        
        input_guardrail = init_supervisor_guardrail(**self.config.guardrail_agents.supervisor_guardrail)
        
        self.support_agent.input_guardrails = [input_guardrail]
        
    
        
        
if __name__=="__main__":
    
    orc = SupportOrchestrator()