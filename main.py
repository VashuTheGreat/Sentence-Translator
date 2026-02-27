from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import os
import sys
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
import asyncio
from fastapi import HTTPException
from dotenv import load_dotenv
import nltk

load_dotenv()

try:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
except Exception as e:
    print(e)

MAX_REQ=int(os.getenv("MAX_REQ", 100))
semaphore=asyncio.Semaphore(MAX_REQ)

# Add project root to path to find src
sys.path.append(os.getcwd())
from src.pipelines.Prediction_Pipeline import PredictionPipeline

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Initialize Prediction Pipeline
prediction_pipeline = PredictionPipeline()

class TranslationRequest(BaseModel):
    data: str

@app.get("/", response_class=HTMLResponse)
@limiter.limit(os.getenv("RATE_LIMIT", "5/minute"))
async def home(request: Request):
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Template Not Found</h1><p>Please ensure templates/index.html exists.</p>"

@app.post("/translate")
@limiter.limit(os.getenv("RATE_LIMIT", "5/minute"))
async def translate_sentence(request: Request, body: TranslationRequest):
    if semaphore.locked() and semaphore._value == 0:
        raise HTTPException(status_code=503, detail="Server Busy. Try again later.")

    async with semaphore:
        sentence = body.data
        if not sentence.strip():
            return {"data": "Please enter a valid sentence."}
        
        result = await prediction_pipeline.initiate_prediction_pipeline(sentence)
        return {"data": result}

if __name__ == "__main__":
    uvicorn.run("main:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)), reload=True)
