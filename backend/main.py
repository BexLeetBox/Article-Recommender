from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query
import pandas as pd
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from backend.utils import load_news_df

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.DataFrame({"id": range(1, 10001), "value": range(10001, 20001)})

news_df = load_news_df()

@app.get("/data/")
def get_data(offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
    print(offset, limit)
    return df.iloc[offset : offset + limit].to_dict(orient="records")

@app.get("/news/")
def get_data(offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
    print(offset, limit)
    return news_df.iloc[offset : offset + limit].to_dict(orient="records")