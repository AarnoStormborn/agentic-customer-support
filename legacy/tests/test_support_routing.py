
"""Test Support Agent routing to SQL Agent."""
import asyncio
import yaml
from google.adk import Runner
import google.adk.sessions as sessions
from google.genai import types as genai_types
from src.agent_team.support_agent import init_support_agent
from src.agent_team.rag_agent import init_rag_agent
from src.agent_team.sql_agent import init_sql_agent
from src.agent_team.web_agent import init_web_agent

async def test_support_agent_routing():
    # Load configuration
    with open("config/agents.yml", "r") as f:
        config = yaml.safe_load(f)
    with open("config/schema.yml", "r") as f:
        schema_yml = f.read()
    
    agents_config = config['agents']
    
    # Initialize sub-agents
    print("Initializing agents...")
    rag_agent = init_rag_agent(agents_config['rag_agent']['name'], agents_config['rag_agent']['prompt'])
    sql_agent = init_sql_agent(agents_config['sql_agent']['name'], agents_config['sql_agent']['prompt'].format(schema=schema_yml))
    web_agent = init_web_agent(agents_config['web_agent']['name'], agents_config['web_agent']['prompt'])
    
    # Initialize Support Agent with sub-agents
    support_agent = init_support_agent(
        name=agents_config['support_agent']['name'],
        prompt=agents_config['support_agent']['prompt'],
        sub_agents=[rag_agent, sql_agent, web_agent]
    )
    print(f"Support Agent initialized with sub-agents: {[a.name for a in support_agent.sub_agents]}\n")
    
    # Setup session
    session_service = sessions.InMemorySessionService()
    await session_service.create_session(app_name="support_test", user_id="user", session_id="session")
    
    runner = Runner(app_name="support_test", agent=support_agent, session_service=session_service)
    
    # Test queries (same as SQL agent test - should route to database_agent)
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
            print("Support Agent Response:")
            response_text = ""
            for event in runner.run(user_id="user", session_id="session", new_message=msg):
                # Track which agent is responding
                if hasattr(event, 'author') and event.author:
                    if event.author != 'customer_support_agent':
                        print(f"  [Routed to: {event.author}]")
                
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            response_text += part.text
            # Truncate long responses
            if len(response_text) > 600:
                print(response_text[:600] + "...")
            else:
                print(response_text)
            print()
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    asyncio.run(test_support_agent_routing())
