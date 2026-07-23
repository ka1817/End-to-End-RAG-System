import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize SQLAlchemy engine
# pool_pre_ping=True handles connection drops on serverless DBs like Neon
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Define the Database Table Schema
class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables in Neon if they don't already exist
def init_db():
    print("Connecting to PostgreSQL (Neon) and verifying tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized successfully!")

# Function to save a user query and AI response
def save_chat_log(question: str, answer: str):
    db = SessionLocal()
    try:
        log_entry = ChatLog(question=question, answer=answer)
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        return log_entry
    except Exception as e:
        db.rollback()
        print(f"Error saving chat log to database: {e}")
    finally:
        db.close()