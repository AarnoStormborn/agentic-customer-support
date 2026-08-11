
"""Test SQL Agent with multiple natural language queries."""
import asyncio
import yaml
from google.adk import Runner
import google.adk.sessions as sessions
from google.genai import types as genai_types
from src.agent_team.sql_agent import init_sql_agent

async def test_sql_agent():
    # Load configuration
    with open("config/agents.yml", "r") as f:
        config = yaml.safe_load(f)
    with open("config/schema.yml", "r") as f:
        schema_yml = f.read()
    
    sql_cfg = config['agents']['sql_agent']
    sql_prompt = sql_cfg['prompt'].format(schema=schema_yml)
    
    # Initialize SQL Agent
    sql_agent = init_sql_agent(sql_cfg['name'], sql_prompt)
    print(f"SQL Agent initialized: {sql_agent.name}\n")
    
    # Setup session
    session_service = sessions.InMemorySessionService()
    await session_service.create_session(app_name="sql_test", user_id="user", session_id="session")
    
    runner = Runner(app_name="sql_test", agent=sql_agent, session_service=session_service)
    
    # Test queries
    test_queries = [
        "How many tickets are there in total?",
        "Show me all critical priority tickets",
        "What are the different ticket types and how many of each?",
        "Find tickets for iPhone purchases with technical issues",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"{'='*60}")
        print(f"Query {i}: {query}")
        print(f"{'='*60}")
        
        msg = genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
        
        try:
            print("SQL Agent Response:")
            response_text = ""
            for event in runner.run(user_id="user", session_id="session", new_message=msg):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            response_text += part.text
            print(response_text[:500] if len(response_text) > 500 else response_text)
            print()
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    asyncio.run(test_sql_agent())
