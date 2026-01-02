""" Ingestion of data into the Database: Tables in Postgres, Text, Embeddings in pgvector """

import os
import openai
import pandas as pd
import typing as T
import asyncio
from pypdf import PdfReader
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from schemas import (
    Document, DocumentChunk, Base
)

import logging

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(levelname)s %(module)s:%(lineno)d - %(message)s"))
console_handler.setLevel(logging.INFO)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), handlers=[console_handler])


def ingest_sql_database(filepath: str, table_name: str, db_conn: str):
    """Synchronous ingestion for CSV data (pandas doesn't support async)"""
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
    
    # OpenAI batch limits (Reduced to prevent timeouts)
    MAX_BATCH_SIZE: int = 100  # Smaller batches for better reliability
    
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
        
        # Async OpenAI client for embedding generation with higher timeout
        self.async_openai_client = openai.AsyncOpenAI(timeout=60.0)
        
        # Convert sync connection string to async (postgresql -> postgresql+asyncpg)
        async_db_conn = db_conn.replace("postgresql://", "postgresql+asyncpg://")
        self.async_engine = create_async_engine(async_db_conn)
        self.async_session_factory = sessionmaker(
            self.async_engine, 
            class_=AsyncSession, 
            expire_on_commit=False
        )

        self.last_chunk_id = 0
        
    def read_pdf(self, file_path: str) -> T.Optional[str]:
        
        try:
            logging.info(f"Reading file: {file_path.split('/')[-1]}")
            reader = PdfReader(file_path)
            text = ""
            
            for i, page in enumerate(reader.pages):
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
                if (i + 1) % 5 == 0:
                     logging.info(f"Processed {i + 1} pages")
                    
            return text
        
        except:
            logging.error(f"Error Reading pdf {file_path}", exc_info=True)
            return None
    
    def split_text(self, text: str) -> T.List[str]:
        
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
            return []

    
    async def generate_embeddings(self, text_chunks: T.List[str]) -> T.List[T.List[float]]:
        """Generate embeddings using async batch inference"""
        try:
            if not text_chunks:
                return []
            
            all_embeddings = []
            
            # Process in batches if exceeding max batch size
            for i in range(0, len(text_chunks), self.MAX_BATCH_SIZE):
                batch = text_chunks[i:i + self.MAX_BATCH_SIZE]
                
                # Async batch inference: single API call per batch
                response = await self.async_openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                
                # Extract embeddings in order (response.data is ordered by index)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            logging.info(f"Async batch embeddings generated: {len(all_embeddings)} embeddings")
            
            return all_embeddings
    
        except:
            logging.error("Error generating embeddings", exc_info=True)
            return []
    
    async def doc_processing_pipeline(self, file_path: str, doc_id: int) -> T.List[DocumentChunk]:
        """Async document processing pipeline"""
        try:
            chunks = []
            
            text = self.read_pdf(file_path=file_path)
            if not text:
                return []
                
            text_chunks = self.split_text(text=text)
            if not text_chunks:
                return []
            
            # Async embedding generation
            embeddings = await self.generate_embeddings(text_chunks=text_chunks)
            if not embeddings:
                return []
            
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
            logging.error("Error in doc processing pipeline", exc_info=True)
            return []
        
    
    async def create_tables(self):
        """Create pgvector extension and tables if they don't exist"""
        async with self.async_engine.begin() as conn:
            # Enable pgvector extension first
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
    
    async def upsert_docs(self) -> bool:
        """Async upsert documents and chunks to the database"""
        try:
            
            docs = []
            doc_chunks = []
            timestamp = datetime.now()
            
            pdf_files = [f for f in os.listdir(self.dir_path) if f.endswith('.pdf')]
            
            for i, pdf_file in enumerate(pdf_files):
                
                logging.info(f"Working on file: {pdf_file}")
                
                doc_obj = {
                    "id": i+1,
                    "doc_name": pdf_file,
                    "created_at": timestamp
                }
                
                file_path = os.path.join(self.dir_path, pdf_file)
                
                # Async pipeline
                chunks = await self.doc_processing_pipeline(
                    file_path=file_path,
                    doc_id=i+1
                )
                
                if not chunks:
                    continue
                    
                docs.append(Document(**doc_obj))
                doc_chunks.extend(chunks)
                
                logging.info(f"Completed processing {pdf_file}")
            
            # Async database session
            async with self.async_session_factory() as session:
                async with session.begin():
                    session.add_all(docs)
                    session.add_all(doc_chunks)
                # commit is automatic when exiting the begin() context
                
            logging.info(f"Session Committed. Upserted {len(docs)} documents and {len(doc_chunks)} chunks")
            return True
                
        except:
            logging.error("Error upserting documents", exc_info=True)
            return False
    
    async def close(self):
        """Cleanup async resources"""
        await self.async_engine.dispose()
            
            
 
if __name__=="__main__":
    
    filepath = "./config/data/tickets.csv"
    table_name = "tickets"    
    db_string = os.getenv("DB_STRING")
    
    # Sync ingestion for CSV (pandas doesn't support async natively)
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
    
    # Run async ingestion pipeline
    async def main():
        try:
            await ingestion.create_tables()
            result = await ingestion.upsert_docs()
            print(f"Complete: {result}")
        finally:
            await ingestion.close()
    
    asyncio.run(main())