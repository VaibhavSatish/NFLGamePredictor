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

def main():
    print("Hello from nflgamepredictor!")


if __name__ == "__main__":
    main()
