
import asyncio
import os
import yaml
from uuid import uuid4
from google.adk import Runner
import google.adk.sessions as sessions
from google.genai import types as genai_types
from src.agent_team.support_agent import init_support_agent
from src.agent_team.rag_agent import init_rag_agent
from src.agent_team.sql_agent import init_sql_agent
from src.agent_team.web_agent import init_web_agent

async def test_back_transfer():
    print("Testing Orchestration Loop (Support -> SQL -> Support)...")
    
    # Load config
    with open("config/agents.yml", "r") as f:
        config = yaml.safe_load(f)
    print("Config loaded.")

    # Schema placeholder
    schema_yml = "{}" 
    # Not needed for logic test, but sql init expects it in prompt format or simplistic.
    # Actually sql init uses prompt.format(schema=schema_yml) in main.py but here I am calling init directly.
    # Let's fix prompt interpolation.
    
    agents_config = config['agents']
    
    # Init Sub-agents
    rag_cfg = agents_config['rag_agent']
    rag_agent = init_rag_agent(rag_cfg['name'], rag_cfg['prompt'])
    
    sql_cfg = agents_config['sql_agent']
    # Use simplistic schema for prompt
    sql_prompt = sql_cfg['prompt'].format(schema="table: tickets") 
    sql_agent = init_sql_agent(sql_cfg['name'], sql_prompt)
    
    web_cfg = agents_config['web_agent']
    web_agent = init_web_agent(web_cfg['name'], web_cfg['prompt'])
    
    # Init Support Agent
    support_cfg = agents_config['support_agent']
    support_agent = init_support_agent(
        name=support_cfg['name'],
        prompt=support_cfg['prompt'],
        sub_agents=[rag_agent, sql_agent, web_agent]
    )
    print("Agents initialized.")
    
    session_service = sessions.InMemorySessionService()
    user_id = "test_user"
    session_id = "test_session_loop"
    app_name = "test_app"
    
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    
    runner = Runner(
        app_name=app_name,
        agent=support_agent,
        session_service=session_service
    )
    
    # Step 1: Ask SQL query
    print("\n--- Sending SQL Query ---")
    msg1 = genai_types.Content(role="user", parts=[genai_types.Part(text="How many tickets in total?")])
    
    sql_answered = False
    transferred_to_sql = False
    transferred_back = False
    
    for event in runner.run(user_id=user_id, session_id=session_id, new_message=msg1):
        # We can inspect logs or just observe the text. 
        # But we want to know if active agent changed.
        # ADK runner state is internal.
        # But if we ask a follow up question that is NON-SQL (e.g. Web), and it works, it means we are back at Support Agent.
        # If we were stuck at SQL Agent, it would fail to answer Web query or try to do SQL on it.
        if event.content and event.content.parts:
             for p in event.content.parts:
                 if p.text:
                     print(f"Response: {p.text}")
                     if "total" in p.text or "tickets" in p.text:
                         sql_answered = True

    print("\n--- Sending Web Query (Check if routed correctly) ---")
    # If control is back at Support Agent, it should route this to Web Agent.
    # If control is stuck at SQL Agent, it will try to use SQL or fail.
    msg2 = genai_types.Content(role="user", parts=[genai_types.Part(text="Search web for 'latest python version'")])
    
    web_answered = False
    
    for event in runner.run(user_id=user_id, session_id=session_id, new_message=msg2):
         if event.content and event.content.parts:
             for p in event.content.parts:
                 if p.text:
                     print(f"Response: {p.text}")
                     if "python" in p.text.lower() or "version" in p.text.lower():
                         web_answered = True
                         
    if sql_answered and web_answered:
        print("\nSUCCESS: System handled SQL query then Web query. Orchestration loop is working.")
    else:
        print(f"\nFAILURE: SQL Answered: {sql_answered}, Web Answered: {web_answered}")
        if not web_answered:
            print("System likely failed to transfer back to Support Agent.")

if __name__ == "__main__":
    asyncio.run(test_back_transfer())
