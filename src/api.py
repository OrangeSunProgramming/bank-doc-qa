"""REST API for the bank document Q&A agent."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agent import Answer, ask

app = FastAPI(title="Bank Document Q&A", version="0.1.0")

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="A question about the 10-K reports")

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/ask", response_model=Answer)
def ask_endpoint(request: QuestionRequest) -> Answer:
    try:
        return ask(request.question)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))