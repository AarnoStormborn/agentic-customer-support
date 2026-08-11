import asyncio
import os
import sys
import yaml
from uuid import uuid4

from google.adk import Runner
import google.adk.sessions as sessions
from google.genai import types as genai_types

from src.agent_team.support_agent import init_support_agent
from src.agent_team.rag_agent import init_rag_agent
from src.agent_team.sql_agent import init_sql_agent
from src.agent_team.web_agent import init_web_agent

async def main():
    print("Agentic Customer Support System")
    print("================================\n")
    
    # Load configuration
    with open("config/agents.yml", "r") as f:
        config = yaml.safe_load(f)
    
    with open("config/schema.yml", "r") as f:
        schema_yml = f.read()
    
    agents_config = config['agents']
    
    # Initialize sub-agents
    print("Initializing agents...")
    
    # RAG Agent (Knowledge Base)
    rag_cfg = agents_config['rag_agent']
    rag_agent = init_rag_agent(rag_cfg['name'], rag_cfg['prompt'])
    print(f"  - {rag_cfg['name']} ready")
    
    # SQL Agent (Database)
    sql_cfg = agents_config['sql_agent']
    sql_prompt = sql_cfg['prompt'].format(schema=schema_yml)
    sql_agent = init_sql_agent(sql_cfg['name'], sql_prompt)
    print(f"  - {sql_cfg['name']} ready")
    
    # Web Agent (Web Search)
    web_cfg = agents_config['web_agent']
    web_agent = init_web_agent(web_cfg['name'], web_cfg['prompt'])
    print(f"  - {web_cfg['name']} ready")
    
    # Initialize Support Agent with sub-agents
    support_cfg = agents_config['support_agent']
    support_agent = init_support_agent(
        name=support_cfg['name'],
        prompt=support_cfg['prompt'],
        sub_agents=[rag_agent, sql_agent, web_agent]
    )
    print(f"  - {support_cfg['name']} ready (supervisor)\n")
    
    # Setup session and runner
    session_service = sessions.InMemorySessionService()
    user_id = "user_" + str(uuid4())[:8]
    session_id = "session_" + str(uuid4())[:8]
    app_name = "customer_support_app"
    
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    
    runner = Runner(
        app_name=app_name,
        agent=support_agent,
        session_service=session_service
    )
    
    print("You are now connected to Customer Support.")
    print("Ask questions about technical manuals, support tickets, or general information.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        msg = genai_types.Content(role="user", parts=[genai_types.Part(text=user_input)])
        
        try:
            print("Support:", end=" ", flush=True)
            for event in runner.run(user_id=user_id, session_id=session_id, new_message=msg):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            print(part.text, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(main())

