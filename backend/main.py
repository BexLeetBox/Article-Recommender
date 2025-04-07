from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi import FastAPI
import data

app = FastAPI()

@app.get("/news")
async def root():
    # This will load data only on first request
    if data.news_data is None:
        data.news_data = data.get_news_data()
    
    # Use data.news_data in your API endpoint
    return {"news_count": len(data.news_data)}