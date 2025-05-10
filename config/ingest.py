""" Ingestion of data into the Database: Tables in Postgres, Text, Embeddings in pgvector """

import os
import openai
import pandas as pd
import typing as T
from pypdf import PdfReader
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from schemas import (
    Document, DocumentChunk
)

import logging

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(levelname)s %(module)s:%(lineno)d - %(message)s"))
console_handler.setLevel(logging.INFO)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), handlers=[console_handler])


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

        self.last_chunk_id = 0
        
    def read_pdf(self, file_path: str):
        
        try:
            logging.info(f"Reading file: {file_path.split('/')[-1]}")
            reader = PdfReader(file_path)
            text = ""
            
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
                    
            return text
        
        except:
            logging.error(f"Error Reading pdf {file_path}", exc_info=True)
    
    def split_text(self, text: str):
        
        try:
            chunks = []
            start = 0
            end = 0
            
            while end < len(text):
                end = start + self.chunk_size
                if end > len(text):
                    end = len(text)
                chunks.append(text[start:end])
                start = end - self.chunk_overlap
                
            logging.info(f"chunks created: {len(chunks)}")
            return chunks
        
        except:
            logging.error("Error splitting text", exc_info=True)

    
    def generate_embeddings(self, text_chunks: T.List[str]):
        
        try:
            embeddings = []
            for chunk in text_chunks:
                response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=chunk
                )
                embeddings.append(response.data[0].embedding)
                
                
            logging.info("embeddings generated")
            
            return embeddings
    
        except:
            logging.error("Error generating embeddings", exc_info=True)
    
    def doc_processing_pipeline(self, file_path: str, doc_id: int):
        
        try:
            chunks = []
            
            text = self.read_pdf(file_path=file_path)
            text_chunks = self.split_text(text=text)
            embeddings = self.generate_embeddings(text_chunks=text_chunks)
            
            for chunk, embedding in zip(text_chunks, embeddings):
                
                self.last_chunk_id += 1
                
                chunk_obj = {
                    "id": self.last_chunk_id,
                    "doc_id": doc_id,
                    "chunk": {
                        "text": chunk,
                        "file": file_path
                    },
                    "embedding": embedding,
                    "created_at": datetime.now()
                }
                
                chunks.append(DocumentChunk(**chunk_obj))
                
            return chunks
        
        except: 
            return None
        
    
    
    def upsert_docs(self):
        
        try:
            
            docs = []
            doc_chunks = []
            timestamp = datetime.now()
            
            for i, pdf_file in enumerate(os.listdir(self.dir_path)):
                
                logging.info(f"Working on file: {pdf_file}")
                
                doc_obj = {
                    "id": i+1,
                    "doc_name": pdf_file,
                    "created_at": timestamp
                }
                # Run the pipeline
                
                file_path = os.path.join(self.dir_path, pdf_file)
                chunks = self.doc_processing_pipeline(
                    file_path=file_path,
                    doc_id=i+1
                )
                
                if not chunks:
                    continue
                    
                docs.append(Document(**doc_obj))
                doc_chunks.extend(chunks)
                
                logging.info("Completed")
                
            with Session(self.db_conn) as session:
                session.add_all(docs)
                session.add_all(doc_chunks)
                session.commit()
                
            logging.info(f"Session Commited. Upserted {len(docs)} documents and {len(chunks)} chunks")
            return True
                
        except:
            logging.error("Error upserting documents", exc_info=True)
            
            
 
if __name__=="__main__":
    
    filepath = "./config/data/tickets.csv"
    table_name = "tickets"    
    db_string = os.getenv("DB_STRING")
    
    ingest_sql_database(
        filepath,
        table_name,
        db_string
    )
    
    print(db_string)
    
    ingestion = RAGIngestion(
        dir_path="./config/data/manuals/",
        chunk_size=1024,
        chunk_overlap=50,
        embedding_model="text-embedding-3-small",
        db_conn=db_string
    )
    
    result = ingestion.upsert_docs()
    
    print(f"Complete: {result}")
    
    
    
    