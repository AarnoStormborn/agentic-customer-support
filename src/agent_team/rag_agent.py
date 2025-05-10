""" RAG / Knowledge Base Agent """

import os
from agents import Agent, function_tool

from sqlalchemy import create_engine, text

from src.utils import generate_embeddings
from src.logger import logger
from src.exception import CustomException

@function_tool
def retriever_tool(query: str, top_k: int):
    
    """
    A function to retrieve information from knowledge base.
    The knowledge base points to a vector database that 
    contains technical manuals
    
    args:
        query: str = The query for which information needs to be retrieved
        top_k: int = Number of search results to retrieve. Increase number if more info is needed
    returns:
        dict: A dictionary containing relevant information along with source
    """
    try: 
        embedding = generate_embeddings(query, model_name=os.getenv("OPENAI_EMBEDDINGS"))
        if not embedding:
            logger.error("Error generating embedding")
            return
        
        logger.info("Query Embedding generated")
        
        db_string = os.getenv("DB_STRING")
        engine = create_engine(db_string)
        
        embedding_str = f"'{str(embedding)}'"
        
        sql_query = f"""
            SELECT id, chunk, created_at
            FROM t_docs_chunks
            ORDER BY embedding <#> '{str(embedding)}'::vector
            LIMIT {top_k}
        """
        
        
        
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            
        logger.info(f"Query Executed. Retrieved {top_k} results.")
        if result:
            response = str([tup[1] for tup in result])
            print(response)
            return response
        else:
            return "No results found"
            

    except Exception as e:
        logger.error(CustomException(e))


def init_rag_agent(name: str, prompt: str) -> Agent:
    
    agent = Agent(
        name=name,
        instructions=prompt,
        tools=[retriever_tool],
        model=os.getenv('OPENAI_MODEL')
    )
    
    return agent
