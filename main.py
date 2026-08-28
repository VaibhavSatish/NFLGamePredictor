import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost 
import nfl_data_py as nfl

app = FastAPI()
model = xgboost.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42)
team_stats = {}

@app.on_event("startup")
def train():
    pass

@app.get("/")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    print("Running")
    uvicorn.run(app, host='127.0.0.1', port=8000)