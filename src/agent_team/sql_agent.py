import os
from agents import Agent, function_tool

from sqlalchemy import create_engine

from src.logger import logger
from src.exception import CustomException


@function_tool
def run_sql_queries(query: str) -> str:
    
    """
    Use this function to run SQL queries on the Tickets Database
    Provide a SQL query to this tool based on the user's request
    
    args:
        query (str): SQL query to run on the DB
    
    returns:
        str: Output of the Query
    """
    
    try:
        
        logger.info("Calling SQL tool")
        
        db_string = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5433/{os.getenv('POSTGRES_DB')}"
        engine = create_engine(db_string)
        
        with engine.connect() as conn:
            result = conn.execute(query)
        
        logger.info("Query Executed")
            
        return str(result) if result else "No results found"
    
    except Exception as e:
        logger.warning(CustomException(e))
        return f"Error executing query: {e}"    
    
    
def init_sql_agent(name: str, prompt: str) -> Agent:
    
    agent = Agent(
        name=name,
        tools=[run_sql_queries],
        instructions=prompt,
        model=os.getenv("OPENAI_MODEL")
    )
    
    return agent

            
            
            
    
