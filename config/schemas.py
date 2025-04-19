from typing import List
from typing import Optional
from datetime import datetime
from sqlalchemy import ForeignKey
from sqlalchemy import String, DateTime
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass    
    
class Document(Base):
    __tablename__ = "t_docs"
    id: Mapped[int] = mapped_column(primary_key=True)
    doc_name: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now())
    
    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, doc_name={self.doc_name!r}, created_at={self.created_at!r})"
    
class DocumentChunk(Base):
    __tablename__ = "t_docs_chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("t_docs.id", ondelete="CASCADE"))
    
    document: Mapped[Document] = relationship("Document", back_populates="chunks")
    
    chunk: Mapped[dict] = mapped_column(JSONB)
    embedding: Mapped[List[float]] = mapped_column(Vector(1536))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now())
        
    