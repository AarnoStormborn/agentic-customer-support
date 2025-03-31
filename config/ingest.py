""" Ingestion of data into the Database: Tables in Postgres, Text, Embeddings in pgvector """

import os
import pandas as pd
from sqlalchemy import create_engine

def ingest_sql_database(filepath: str, table_name: str, db_conn: str):
    
    try:
        engine = create_engine(db_conn)
        df = pd.read_csv(filepath)
        df.to_sql(table_name, engine, if_exists='replace', index=False)

        print(f"Data successfully loaded into '{table_name}' table.")

    except Exception as e:
        print(f"An error occurred: {e}")
        
        
def ingest_text_data(dirpath: str):
    
    # load all pdf files
    # collect all texts
    # preprocess text
    # create embeddings
    # insert into pgvector
    
    pass
 
 
if __name__=="__main__":
    
    filepath = "./config/data/tickets.csv"
    table_name = "tickets"    
    db_string = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5433/{os.getenv('POSTGRES_DB')}"
    
    ingest_sql_database(
        filepath,
        table_name,
        db_string
    )
    
    
    
    