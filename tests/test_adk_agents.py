
import asyncio
import os
import yaml
from uuid import uuid4
from google.adk import Runner
import google.adk.sessions as sessions
from google.genai import types as genai_types

from src.agent_team.rag_agent import init_rag_agent
from src.agent_team.sql_agent import init_sql_agent
from src.agent_team.web_agent import init_web_agent

async def run_agent_test(agent_name, agent_init_func, prompt, test_query):
    print(f"--- Testing {agent_name} ---")
    try:
        agent = agent_init_func("test_" + agent_name, prompt)
        
        session_service = sessions.InMemorySessionService()
        user_id = "user_test"
        session_id = f"session_{agent_name}"
        app_name = "test_app"
        
        await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
        
        runner = Runner(
            app_name=app_name,
            agent=agent,
            session_service=session_service
        )
        
        msg = genai_types.Content(role="user", parts=[genai_types.Part(text=test_query)])
        
        print(f"User: {test_query}")
        print("Agent:", end=" ", flush=True)
        
        response_text = ""
        for event in runner.run(user_id=user_id, session_id=session_id, new_message=msg):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(part.text, end="", flush=True)
                        response_text += part.text
        print("\n")
        
        if not response_text:
            print(f"FAILED: No response from {agent_name}")
        else:
            print(f"SUCCESS: {agent_name} responded")
            
    except Exception as e:
        print(f"FAILED: Error testing {agent_name}: {e}")
        import traceback
        traceback.print_exc()

async def main():
    # Load configuration
    with open("config/agents.yml", "r") as f:
        config = yaml.safe_load(f)
    agents_config = config['agents']

    # Test Web Agent
    web_cfg = agents_config['web_agent']
    # await run_agent_test("Web_Agent", init_web_agent, web_cfg['prompt'], 
    #                      "Who is the CEO of Google?")

    # Test SQL Agent
    # Note: This might fail if DB is not set up, but we want to verify the agent/tool wiring
    with open("config/schema.yml", "r") as f:
        schema_yml = f.read()
    sql_cfg = agents_config['sql_agent']
    sql_prompt = sql_cfg['prompt'].format(schema=schema_yml)
    # await run_agent_test("SQL_Agent", init_sql_agent, sql_prompt, 
    #                      "Count the number of tickets.")

    # Test RAG Agent
    # Note: RAG requires embeddings and DB.
    rag_cfg = agents_config['rag_agent']
    await run_agent_test("RAG_Agent", init_rag_agent, rag_cfg['prompt'], 
                         "How do I reset my router?")

if __name__ == "__main__":
    asyncio.run(main())
