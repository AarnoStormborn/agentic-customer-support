"""
Conversation Simulation Script
Runs multiple test conversations engaging different agents and tools.
Produces results.json with conversation outcomes.
"""
import asyncio
import json
import os
import re
import sys
import yaml
from uuid import uuid4
from datetime import datetime
from google.adk import Runner
import google.adk.sessions as sessions
from google.genai import types as genai_types

from src.agent_team.support_agent import init_support_agent
from src.agent_team.rag_agent import init_rag_agent
from src.agent_team.sql_agent import init_sql_agent
from src.agent_team.web_agent import init_web_agent

# Capture logs to track agent/tool usage
import logging

class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
        
    def emit(self, record):
        self.records.append(record.getMessage())
        
    def get_and_clear(self):
        records = self.records.copy()
        self.records = []
        return records

log_capture = LogCapture()
log_capture.setLevel(logging.INFO)

# Add to our logger
from src.logger import logger
logger.addHandler(log_capture)


def parse_agent_and_tools(log_messages):
    """Extract agents and tools from log messages."""
    agents_used = set()
    tools_used = []
    
    for msg in log_messages:
        # Pattern: Agent 'agent_name' calling tool: tool_name with args: {...}
        match = re.search(r"Agent '([^']+)' calling tool: (\w+) with args: (.+)", msg)
        if match:
            agent_name = match.group(1)
            tool_name = match.group(2)
            args_str = match.group(3)
            agents_used.add(agent_name)
            tools_used.append({
                "agent": agent_name,
                "tool": tool_name,
                "args": args_str
            })
    
    return list(agents_used), tools_used


async def run_conversation(runner, user_id, session_id, queries):
    """Run a multi-turn conversation and capture results."""
    conversation_log = []
    
    for query in queries:
        # Clear previous logs
        log_capture.get_and_clear()
        
        msg = genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
        
        response_text = ""
        for event in runner.run(user_id=user_id, session_id=session_id, new_message=msg):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_text += part.text
        
        # Get logs for this turn
        logs = log_capture.get_and_clear()
        agents, tools = parse_agent_and_tools(logs)
        
        conversation_log.append({
            "query": query,
            "response": response_text,
            "agents_involved": agents,
            "tools_used": tools
        })
    
    return conversation_log


async def main():
    print("=" * 60)
    print("Agent Conversation Simulation")
    print("=" * 60)
    
    # Load configuration
    with open("config/agents.yml", "r") as f:
        config = yaml.safe_load(f)
    
    with open("config/schema.yml", "r") as f:
        schema_yml = f.read()
    
    agents_config = config['agents']
    
    # Initialize agents
    print("Initializing agents...")
    
    rag_cfg = agents_config['rag_agent']
    rag_agent = init_rag_agent(rag_cfg['name'], rag_cfg['prompt'])
    
    sql_cfg = agents_config['sql_agent']
    sql_prompt = sql_cfg['prompt'].format(schema=schema_yml)
    sql_agent = init_sql_agent(sql_cfg['name'], sql_prompt)
    
    web_cfg = agents_config['web_agent']
    web_agent = init_web_agent(web_cfg['name'], web_cfg['prompt'])
    
    support_cfg = agents_config['support_agent']
    support_agent = init_support_agent(
        name=support_cfg['name'],
        prompt=support_cfg['prompt'],
        sub_agents=[rag_agent, sql_agent, web_agent]
    )
    
    print("Agents ready.\n")
    
    # Define test conversations - longer and product-specific
    test_conversations = [
        {
            "name": "Database Query Conversation",
            "description": "Testing SQL Agent with ticket-related queries (5 turns)",
            "queries": [
                "How many total tickets are there in the database?",
                "Show me tickets with critical priority",
                "What are the different ticket types we have?",
                "List all tickets created in the last month",
                "Which tickets are currently unresolved?"
            ]
        },
        {
            "name": "Web Search Conversation",
            "description": "Testing Web Agent with internet search queries (4 turns)",
            "queries": [
                "Search the web for the latest AI news in 2026",
                "Find information about Google Gemini latest updates",
                "What are the best practices for customer support automation?",
                "Search for common issues with smart home devices"
            ]
        },
        {
            "name": "Knowledge Base Conversation - Mobile Devices",
            "description": "Testing RAG Agent with Google Pixel, iPhone, and Xperia queries",
            "queries": [
                "How do I take a screenshot on a Google Pixel phone?",
                "What are the battery saving tips for iPhone?",
                "How do I transfer data from my old phone to a Sony Xperia?",
                "What is the camera setup process on Google Pixel?",
                "How do I reset my iPhone to factory settings?"
            ]
        },
        {
            "name": "Knowledge Base Conversation - Computers",
            "description": "Testing RAG Agent with Dell XPS and HP Pavilion queries",
            "queries": [
                "How do I update the BIOS on a Dell XPS 15?",
                "What are the troubleshooting steps if my HP Pavilion won't boot?",
                "How do I connect an external monitor to Dell XPS 15?",
                "What is the recommended maintenance for HP Pavilion laptops?"
            ]
        },
        {
            "name": "Multi-Agent Conversation",
            "description": "Testing routing across all agents in one session (6 turns)",
            "queries": [
                "How many open tickets do we have?",
                "Search the web for latest smartphone troubleshooting guides",
                "How do I setup my new Google Pixel?",
                "What are the most common ticket categories?",
                "Find information about Dell XPS 15 specifications online",
                "How do I connect Bluetooth devices to my HP Pavilion?"
            ]
        }
    ]
    
    results = {
        "simulation_timestamp": datetime.now().isoformat(),
        "model_used": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "conversations": []
    }
    
    for i, conv_def in enumerate(test_conversations):
        print(f"\n{'='*60}")
        print(f"Conversation {i+1}: {conv_def['name']}")
        print(f"Description: {conv_def['description']}")
        print("="*60)
        
        # Create fresh session for each conversation
        session_service = sessions.InMemorySessionService()
        user_id = f"sim_user_{i}"
        session_id = f"sim_session_{i}_{uuid4().hex[:8]}"
        app_name = "simulation_app"
        
        await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
        
        runner = Runner(
            app_name=app_name,
            agent=support_agent,
            session_service=session_service
        )
        
        # Run conversation
        conv_log = await run_conversation(runner, user_id, session_id, conv_def['queries'])
        
        # Print summary
        for turn in conv_log:
            print(f"\n[User]: {turn['query']}")
            print(f"[Response]: {turn['response'][:200]}..." if len(turn['response']) > 200 else f"[Response]: {turn['response']}")
            print(f"[Agents]: {', '.join(turn['agents_involved']) if turn['agents_involved'] else 'None detected'}")
            print(f"[Tools]: {[t['tool'] for t in turn['tools_used']] if turn['tools_used'] else 'None detected'}")
        
        results["conversations"].append({
            "conversation_id": i + 1,
            "name": conv_def['name'],
            "description": conv_def['description'],
            "turns": conv_log,
            "summary": {
                "total_turns": len(conv_log),
                "unique_agents": list(set(a for turn in conv_log for a in turn['agents_involved'])),
                "unique_tools": list(set(t['tool'] for turn in conv_log for t in turn['tools_used']))
            }
        })
    
    # Write results to JSON
    output_path = "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Simulation Complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print overall summary
    all_agents = set()
    all_tools = set()
    for conv in results["conversations"]:
        all_agents.update(conv["summary"]["unique_agents"])
        all_tools.update(conv["summary"]["unique_tools"])
    
    print(f"\nOverall Summary:")
    print(f"  Total Conversations: {len(results['conversations'])}")
    print(f"  Unique Agents Used: {all_agents}")
    print(f"  Unique Tools Used: {all_tools}")


if __name__ == "__main__":
    asyncio.run(main())
