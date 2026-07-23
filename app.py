from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.vectordb import get_retriever
from src.retrieval import get_rag_chain
from database import init_db, save_chat_log  # <-- Updated import path

rag_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Pinecone Retriever and Groq RAG chain...")
    try:
        # 1. Initialize PostgreSQL tables
        init_db()

        # 2. Load Retriever & RAG Chain
        retriever = get_retriever()
        rag_chain = get_rag_chain(retriever)
        rag_state["chain"] = rag_chain
        print("RAG Pipeline and Database connection successfully loaded!")
    except Exception as e:
        print(f"Error during startup initialization: {e}")
        raise e
        
    yield
    print("Shutting down application...")
    rag_state.clear()

app = FastAPI(
    title="10th Social Textbook RAG API",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str

@app.get("/", response_class=HTMLResponse)
async def get_web_ui(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request}
    )

@app.post("/query", response_model=QueryResponse)
async def query_textbook(request: QueryRequest):
    question = request.question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    rag_chain = rag_state.get("chain")
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG Chain is not initialized.")
        
    try:
        # 1. Generate answer from LLM
        answer = rag_chain.invoke(question)

        # 2. Log Question and Answer into Neon PostgreSQL DB
        save_chat_log(question=question, answer=answer)

        return QueryResponse(question=question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")