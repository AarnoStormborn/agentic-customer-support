""" Ingestion of data into the Database: Tables in Postgres, Text, Embeddings in pgvector """

import os
import openai
import pandas as pd
from pypdf import PDFReader
from datetime import datetime

from sqlalchemy import create_engine, insert
from sqlalchemy.orm import Session

from schemas import (
    Document, DocumentChunk
)

def ingest_sql_database(filepath: str, table_name: str, db_conn: str):
    
    try:
        engine = create_engine(db_conn)
        df = pd.read_csv(filepath)
        df.to_sql(table_name, engine, if_exists='replace', index=False)

        print(f"Data successfully loaded into '{table_name}' table.")

    except Exception as e:
        print(f"An error occurred: {e}")
        
    
class RAGIngestion:
    
    document_table: str = "t_docs"
    embedding_table: str = "t_docs_chunks"
    
    def __init__(
        self,
        dir_path: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        db_conn: str
    ):
        self.dir_path = dir_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.db_conn = create_engine(db_conn)
        
    def read_pdf(self, file_path):
        
        reader = PDFReader(file_path)
        text = ""
        
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
                
        return text
    
    def split_text(self, text):
        
        chunks = []
        start = 0
        end = 0
        
        while end < len(text):
            end = start + self.chunk_size
            if end > len(text):
                end = len(text)
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            
        return chunks
    
    def generate_embeddings(self, text_chunks):
        embeddings = []
        for chunk in text_chunks:
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=chunk
            )
            embeddings.append(response['data'][0]['embedding'])
        
        return embeddings
    
    
    def upsert_docs(self):
        
        docs = []
        timestamp = datetime.now()
        
        for i, pdf_file in enumerate(os.listdir(self.dir_path)):
            
            doc_obj = {
                "id": i+1,
                "doc_name": pdf_file,
                "created_at": timestamp
            }
            docs.append(doc_obj)
            
        with Session(self.db_conn) as session:
            session.execute(
                insert(Document),
                docs,
            )     
            
    def upsert_doc_chunks(self):
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
    
    
    
    